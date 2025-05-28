import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from sglang.srt.distributed import GroupCoordinator, patch_tensor_parallel_group
from sglang.srt.layers.dp_attention import disable_dp_size
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    SpecReqMigrationInfo,
    assign_draft_cache_locs,
    fast_topk,
    select_top_k_tokens,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import empty_context, get_available_gpu_memory, is_cuda_available

if is_cuda_available():
    from sgl_kernel import segment_packbits

logger = logging.getLogger(__name__)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with disable_dp_size(), patch_tensor_parallel_group(tp_group):
        yield

class EagleWorkerLogger:
    def __init__(self, log_file_path: str = "eagle_worker_timings_all_settings.csv"):
        """
        Initializes the logger.

        Args:
            log_file_path: Path to the CSV file where timings will be logged.
        """
        self.log_file_path = log_file_path
        self.fieldnames = [
            "timestamp",
            "speculative_num_steps",
            "top_k",
            "num_accept_tokens",
            "batch_size",
            "draft_time_ms",
            "verify_time_ms",
            "extend_after_decode_time_ms",
            "total_forward_batch_spec_time_ms"
        ]
        self._initialize_log_file()

    def _initialize_log_file(self):
        """
        Creates the log file and writes the header if it doesn't exist or is empty.
        """
        write_header = not os.path.exists(self.log_file_path) or os.path.getsize(self.log_file_path) == 0
        try:
            with open(self.log_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if write_header:
                    writer.writeheader()
        except IOError as e:
            print(f"Error initializing log file {self.log_file_path}: {e}")
            # Potentially raise or handle more gracefully depending on requirements

    def log_timings(
        self,
        num_draft_tokens : int,
        top_k : int,
        num_accept_tokens : int,
        batch_size: int,
        draft_time: Optional[float],
        verify_time: Optional[float],
        extend_after_decode_time: Optional[float],
        total_time: float
    ):
        """
        Logs the timings for one step of speculative generation.

        Args:
            batch_size: The current batch size.
            draft_time: Time taken for the draft phase (in seconds). None if not applicable.
            verify_time: Time taken for the verify phase (in seconds). None if not applicable.
            extend_after_decode_time: Time for forward_draft_extend_after_decode (in seconds). None if not applicable.
            total_time: Total time for the forward_batch_speculative_generation call (in seconds).
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "speculative_num_steps" : num_draft_tokens,
            "top_k" : top_k,
            "num_accept_tokens" : num_accept_tokens,
            "batch_size": batch_size,
            "draft_time_ms": (draft_time * 1000) if draft_time is not None else 0.0,
            "verify_time_ms": (verify_time * 1000) if verify_time is not None else 0.0,
            "extend_after_decode_time_ms": (extend_after_decode_time * 1000) if extend_after_decode_time is not None else 0.0,
            "total_forward_batch_spec_time_ms": total_time * 1000
        }
        try:
            with open(self.log_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(log_entry)
        except IOError as e:
            print(f"Error writing to log file {self.log_file_path}: {e}")



class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.padded_static_len = self.speculative_num_steps + 1
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override context length with target model's context length
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with empty_context():
            super().__init__(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                server_args=server_args,
                nccl_port=nccl_port,
                dp_rank=dp_rank,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 models don't share lm_head
            self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            self.hot_token_id = self.draft_model_runner.model.get_hot_token_id().to(
                embed.device
            )
        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(self.draft_model_runner.tp_group):
            self.init_attention_backend()
            self.init_cuda_graphs()

        
        self.tmp_migration_requests = None
        
        self.migrate_out_steps: List[int] = [2, 6, 10, 14, 18]
        self.migrate_in_steps: List[int] = [4, 8, 12, 16, 20]

        self.cur_decoding_step = 0
        self.enable_migration_test = True
        self.log_accept_length_file = "accept_length.txt"
    
    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        if self.server_args.attention_backend == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = True
        elif self.server_args.attention_backend == "triton":
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = False
        elif self.server_args.attention_backend == "flashinfer_mla":
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = True
        else:
            raise ValueError(
                f"EAGLE is not supportted in attention backend {self.server_args.attention_backend}"
            )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # Capture draft
        tic = time.time()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph end. Time elapsed: {time.time() - tic:.2f} s. avail mem={after_mem:.2f} GB. mem usage={(before_mem - after_mem):.2f} GB."
        )

        # Capture extend
        if self.draft_extend_attn_backend:
            raise NotImplementedError()

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int]:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepeted,
            the batch id (used for overlap schedule), and number of accepeted tokens.
        """
        if batch.forward_mode.is_decode():
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                spec_info, to_free_cache_loc = self.draft(batch)
            logits_output, verify_output, model_worker_batch = self.verify(
                batch, spec_info
            )

            # Free cache loc (we put it here to avoid synchronization and hide kernel launch overhead.)
            self.token_to_kv_pool_allocator.free(to_free_cache_loc)

            # If it is None, it means all requests are finished
            if batch.spec_info.verified_id is not None:
                with self.draft_tp_context(self.draft_model_runner.tp_group):
                    self.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                verify_output.verified_id,
                model_worker_batch.bid,
                sum(verify_output.accept_length_per_req_cpu),
            )
        elif batch.forward_mode.is_idle():
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids, _ = (
                self.target_worker.forward_batch_generation(
                    ForwardBatch.init_new(
                        model_worker_batch, self.target_worker.model_runner
                    )
                )
            )
            return logits_output, next_token_ids, model_worker_batch.bid, 0, False
        else:
            logits_output, next_token_ids, bid = self.forward_target_extend(batch)
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids
                )
            return logits_output, next_token_ids, bid, 0

    def test_migrate_forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int]:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed
        the final output batch doesn't have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepeted,
            the batch id (used for overlap schedule), and number of accepeted tokens.
        """
        assert not batch.spec_algorithm.is_none()
        if batch.forward_mode.is_decode():
            if self.enable_migration_test:
                if batch.spec_info.accept_length is not None:
                    accept_lengths_to_log = str(batch.spec_info.accept_length.tolist())
                    log_line = f"decode_step: {self.cur_decoding_step}, batch_size: {batch.batch_size()}, accept_lengths_this_step: {accept_lengths_to_log}\n"
                    with open(self.log_accept_length_file, 'a') as f:
                        f.write(log_line)
                if self.cur_decoding_step in self.migrate_out_steps:
                    self.tmp_migration_requests = self.send_req_migrate(batch, [0,1])
                else:
                    if self.cur_decoding_step in self.migrate_in_steps:
                        self.tmp_migration_requests = self.recv_req_migrate(batch, self.tmp_migration_requests)
                self.cur_decoding_step += 1
            
            t_start = time.time()
            spec_info, to_free_cache_loc = self.draft(batch)
            torch.cuda.synchronize()
            t_draft_end = time.time()
            logits_output, verify_output, model_worker_batch = self.verify(
                batch, spec_info
            )
            torch.cuda.synchronize()
            t_verify_end = time.time()
            # Free cache loc (we put it here to avoid synchronization and hide kernel launch overhead.)
            self.token_to_kv_pool_allocator.free(to_free_cache_loc)
            # if it is None, means all requests are finished
            t_draft_extend_start = time.time()
            if batch.spec_info.verified_id is not None:
                self.forward_draft_extend_after_decode(batch)
            torch.cuda.synchronize()
            t_end = time.time()
            
            #self.time_logger.log_timings(
            #    self.speculative_num_steps,
            #    self.topk,
            #    sum(verify_output.accept_length_per_req_cpu) / len(verify_output.accept_length_per_req_cpu) + 1,
            #    batch.batch_size(),
            #    t_draft_end - t_start,
            #    t_verify_end - t_draft_end,
            #    t_end - t_draft_extend_start,
            #    t_end - t_start
            #)
            return (
                logits_output,
                verify_output.verified_id,
                model_worker_batch.bid,
                sum(verify_output.accept_length_per_req_cpu),
            )

        else:
            logits_output, next_token_ids, bid = self.forward_target_extend(batch)
            self.forward_draft_extend(
                batch, logits_output.hidden_states, next_token_ids
            )
            return logits_output, next_token_ids, bid, 0
    
    
    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            bid: The model batch ID. Used for overlap schedule.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids = self.target_worker.forward_batch_generation(
            model_worker_batch
        )
        return logits_output, next_token_ids, model_worker_batch.bid

    def draft(self, batch: ScheduleBatch):
        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        out_cache_loc = batch.alloc_token_slots(
            num_seqs * self.topk * self.speculative_num_steps
        )
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
        )
        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

        # Get forward batch
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_cuda_graph:
            score_list, token_list, parents_list = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            # Initialize attention backend
            self.draft_attn_backend.init_forward_metadata(forward_batch)
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            # Run forward steps
            score_list, token_list, parents_list = self.draft_forward(forward_batch)

        ret = EagleVerifyInput.create(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
        )
        return ret, out_cache_loc

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            out_cache_loc = out_cache_loc.view(forward_batch.batch_size, -1)
            forward_batch.out_cache_loc = out_cache_loc[
                :, self.topk * i : self.topk * (i + 1)
            ].flatten()
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        return score_list, token_list, parents_list

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        spec_info.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, _ = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True
        )
        self._detect_nan_if_needed(logits_output)
        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch, logits_output, self.token_to_kv_pool_allocator
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepeted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepeted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepeted_indices]

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = res.draft_input

        if batch.return_logprob:
            self.add_logprob_values(batch, res, logits_output)

        return logits_output, res, model_worker_batch

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums_repeat_interleaved)

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(logprobs, token_ids_logprobs_repeat_interleaved)

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: List[int],
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
        )
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        logits_output = self.draft_model_runner.forward(forward_batch)
        self._detect_nan_if_needed(logits_output)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        # Backup fileds that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob

        # Prepare metadata
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.return_logprob = False
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )

        # Run
        logits_output = self.draft_model_runner.forward(forward_batch)

        self._detect_nan_if_needed(logits_output)
        self.capture_for_decode(logits_output, forward_batch.spec_info)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")

    def send_req_migrate(self, batch: ScheduleBatch, migration_indices: List[int]) -> List[SpecReqMigrationInfo]:
        if not migration_indices:
            return []

        migrated_req_infos: List[SpecReqMigrationInfo] = []
        
        # Sort indices in descending order to avoid issues when removing items from lists/tensors
        # by index.
        sorted_migration_indices = sorted(list(set(migration_indices)), reverse=True)

        # --- 1. Gather information and KV cache for requests to be migrated ---
        req_pool_indices_to_migrate = []
        slots_to_free_flat_list = []
        
        # Assuming self.model_runner is the draft_model_runner
        num_draft_layers = self.model_runner.model_config.num_hidden_layers
        num_target_layers = self.target_worker.model_runner.model_config.num_hidden_layers
        
        # Prepare to collect K and V caches for all layers for all migrating requests
        # k_caches_for_migration_all_requests[req_mig_idx][layer_idx]
        draft_k_caches_for_migration_all_requests: List[List[torch.Tensor]] = [[] for _ in sorted_migration_indices]
        draft_v_caches_for_migration_all_requests: List[List[torch.Tensor]] = [[] for _ in sorted_migration_indices]
        target_k_caches_for_migration_all_requests: List[List[torch.Tensor]] = [[] for _ in sorted_migration_indices]
        target_v_caches_for_migration_all_requests: List[List[torch.Tensor]] = [[] for _ in sorted_migration_indices]

        # Get physical K/V buffers for all layers once
        draft_physical_k_buffers_all_layers = [self.model_runner.token_to_kv_pool.get_key_buffer(l) for l in range(num_draft_layers)]
        draft_physical_v_buffers_all_layers = [self.model_runner.token_to_kv_pool.get_value_buffer(l) for l in range(num_draft_layers)]
        target_physical_k_buffers_all_layers = [self.target_worker.model_runner.token_to_kv_pool.get_key_buffer(l) for l in range(num_target_layers)]
        target_physical_v_buffers_all_layers = [self.target_worker.model_runner.token_to_kv_pool.get_value_buffer(l) for l in range(num_target_layers)]
        
        for i, batch_idx in enumerate(sorted_migration_indices):
            if not (0 <= batch_idx < len(batch.reqs)):
                logger.warning(f"SendReqMigrate: Invalid batch_idx {batch_idx} for batch size {len(batch.reqs)}. Skipping.")
                # Adjust k_caches... lists if we skip
                draft_k_caches_for_migration_all_requests.pop(i)
                draft_v_caches_for_migration_all_requests.pop(i)
                target_k_caches_for_migration_all_requests.pop(i)
                target_v_caches_for_migration_all_requests.pop(i)
                continue

            req_to_migrate: Req = batch.reqs[batch_idx]
            req_pool_idx = batch.req_pool_indices[batch_idx].item()
            current_seq_len = batch.seq_lens[batch_idx].item()

            # a. Extract KV Cache
            kv_cache_len_for_req = current_seq_len
            target_kvcache_k_for_req_all_layers: List[torch.Tensor] = []
            target_kvcache_v_for_req_all_layers: List[torch.Tensor] = []
            draft_kvcache_k_for_req_all_layers: List[torch.Tensor] = []
            draft_kvcache_v_for_req_all_layers: List[torch.Tensor] = []

            if current_seq_len > 0:
                # Get physical slot indices for this request's KV cache
                # shape: [current_seq_len]
                physical_slots_for_req = self.req_to_token_pool.req_to_token[req_pool_idx, :current_seq_len].clone()
                slots_to_free_flat_list.append(physical_slots_for_req)

                for layer_idx in range(num_target_layers):
                    k_physical_buffer = target_physical_k_buffers_all_layers[layer_idx]
                    v_physical_buffer = target_physical_v_buffers_all_layers[layer_idx]

                    # Gather K and V for this layer for this request
                    # shape: [current_seq_len, num_kv_heads, head_dim]
                    k_for_layer_req = torch.index_select(k_physical_buffer, 0, physical_slots_for_req)
                    v_for_layer_req = torch.index_select(v_physical_buffer, 0, physical_slots_for_req)
                    
                    target_kvcache_k_for_req_all_layers.append(k_for_layer_req)
                    target_kvcache_v_for_req_all_layers.append(v_for_layer_req)
                target_k_caches_for_migration_all_requests[i] = target_kvcache_k_for_req_all_layers
                target_v_caches_for_migration_all_requests[i] = target_kvcache_v_for_req_all_layers
                    
                for layer_idx in range(num_draft_layers):
                    k_physical_buffer = draft_physical_k_buffers_all_layers[layer_idx]
                    v_physical_buffer = draft_physical_v_buffers_all_layers[layer_idx]

                    # Gather K and V for this layer for this request
                    # shape: [current_seq_len, num_kv_heads, head_dim]
                    k_for_layer_req = torch.index_select(k_physical_buffer, 0, physical_slots_for_req)
                    v_for_layer_req = torch.index_select(v_physical_buffer, 0, physical_slots_for_req)

                    draft_kvcache_k_for_req_all_layers.append(k_for_layer_req)
                    draft_kvcache_v_for_req_all_layers.append(v_for_layer_req)

                draft_k_caches_for_migration_all_requests[i] = draft_kvcache_k_for_req_all_layers
                draft_v_caches_for_migration_all_requests[i] = draft_kvcache_v_for_req_all_layers


            # b. spec_info extract

            current_topk_p_list = batch.spec_info.topk_p[batch_idx].tolist()
            current_topk_idx_list = batch.spec_info.topk_index[batch_idx].tolist()
            current_verified_id_int = batch.spec_info.verified_id[batch_idx].item()
            current_hidden_state = batch.spec_info.hidden_states[batch_idx].clone()
            # c. Create SpecReqMigrationInfo
            # Note: target_kvcache in SpecReqMigrationInfo is defined as List[torch.Tensor]
            # where each tensor is (num_layers, kvcache_len, num_heads, head_dim).
            # We have k_caches_for_migration[i] as List[Tensor(kv_len, num_heads, head_dim)] (per layer)
            # We need to stack them or adjust the data structure definition.
            # Assuming SpecReqMigrationInfo expects separate K and V lists:
            mig_info = SpecReqMigrationInfo(
                target_model_kcache=target_k_caches_for_migration_all_requests[i],
                target_model_vcache=target_v_caches_for_migration_all_requests[i],
                draft_model_kcache=draft_k_caches_for_migration_all_requests[i],
                draft_model_vcache=draft_v_caches_for_migration_all_requests[i],
                last_hidden_state=current_hidden_state,
                kvcache_len=kv_cache_len_for_req,
                request=req_to_migrate,
                topk_p=current_topk_p_list,
                topk_index=current_topk_idx_list,
                verified_id=current_verified_id_int
            )
            migrated_req_infos.append(mig_info)
            req_pool_indices_to_migrate.append(req_pool_idx)

        # --- 2. Free KV cache from token_to_kv_pool_allocator ---
        if slots_to_free_flat_list:
            all_slots_to_free_flat = torch.cat(slots_to_free_flat_list)
            if all_slots_to_free_flat.numel() > 0:
                self.token_to_kv_pool_allocator.free(all_slots_to_free_flat)

        # --- 3. Free request slots from req_to_token_pool ---
        # This marks the req_pool_idx as available.
        # req_to_token_pool usually has a .free(req_pool_idx_tensor) method
        if req_pool_indices_to_migrate:
            req_pool_indices_to_free_tensor = torch.tensor(req_pool_indices_to_migrate, device=self.device, dtype=torch.long)
            self.req_to_token_pool.free(req_pool_indices_to_free_tensor)

        # --- 4. Remove migrated requests and their data from the current batch ---
        # We need a robust way to filter the ScheduleBatch object.
        # If ScheduleBatch has a method like `filter_batch_by_indices_to_keep`, that's ideal.
        # Otherwise, we manually recreate/filter its components.

        if sorted_migration_indices: # Only if there's something to remove
            # Create a mask of requests to keep
            num_original_reqs = len(batch.reqs)
            keep_mask = [True] * num_original_reqs
            for batch_idx in sorted_migration_indices: # sorted_migration_indices is already reversed
                if 0 <= batch_idx < num_original_reqs:
                    keep_mask[batch_idx] = False
            
            indices_to_keep = [i for i, keep in enumerate(keep_mask) if keep]


            indices_to_keep_tensor = torch.tensor(indices_to_keep, device=self.device, dtype=torch.long)
            batch.reqs = [batch.reqs[i] for i in indices_to_keep]
            
            if batch.req_pool_indices is not None:
                batch.req_pool_indices = batch.req_pool_indices[indices_to_keep_tensor]
            if batch.seq_lens is not None:
                batch.seq_lens = batch.seq_lens[indices_to_keep_tensor]
            
            # For input_ids and out_cache_loc, if they are flattened representations
            # of all tokens in the batch, filtering is more complex.
            # If they are per-request (e.g., after a prepare_for_decode), then simple indexing might work.
            # This part is tricky and depends on the current state of these tensors.
            # Assuming forward_mode is DECODE or after verify, where input_ids/out_cache_loc are per-request (batch_size, 1) or (batch_size)

            batch.input_ids = batch.input_ids[indices_to_keep_tensor]
            batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
 

            batch.spec_info.topk_p = batch.spec_info.topk_p[indices_to_keep_tensor]
            batch.spec_info.topk_index = batch.spec_info.topk_index[indices_to_keep_tensor]
            batch.spec_info.verified_id = batch.spec_info.verified_id[indices_to_keep_tensor]
            batch.spec_info.hidden_states = batch.spec_info.hidden_states[indices_to_keep_tensor]

            # Filter other per-request lists/tensors in ScheduleBatch if they exist and are populated
            if batch.prefix_lens is not None and len(batch.prefix_lens) == num_original_reqs:
                batch.prefix_lens = [batch.prefix_lens[i] for i in indices_to_keep]
            if batch.extend_lens is not None and len(batch.extend_lens) == num_original_reqs:
                batch.extend_lens = [batch.extend_lens[i] for i in indices_to_keep]
            if batch.extend_logprob_start_lens is not None and len(batch.extend_logprob_start_lens) == num_original_reqs:
                batch.extend_logprob_start_lens = [batch.extend_logprob_start_lens[i] for i in indices_to_keep]
            
            batch.sampling_info.temperatures = batch.sampling_info.temperatures[indices_to_keep_tensor]
            batch.sampling_info.top_ps = batch.sampling_info.top_ps[indices_to_keep_tensor]
            batch.sampling_info.top_ks = batch.sampling_info.top_ks[indices_to_keep_tensor]
            batch.sampling_info.min_ps = batch.sampling_info.min_ps[indices_to_keep_tensor]

            logger.info(f"SpecReqMigrate: Migrated {len(sorted_migration_indices)} requests. Batch size now {len(batch.reqs)}.")

        return migrated_req_infos

    def recv_req_migrate(self, batch: ScheduleBatch, migration_infos: List[SpecReqMigrationInfo]):
        """
        Receives migrated request information and adds/merges them into the current ScheduleBatch.
        """
        if not migration_infos:
            return

        num_migrated_reqs = len(migration_infos)
        logger.info(f"recv_req_migrate: Receiving {num_migrated_reqs} requests.")

        # --- 1. Allocate resources for new requests ---
        # a. Allocate request pool indices
        new_req_pool_indices = self.req_to_token_pool.alloc(num_migrated_reqs)
        new_req_pool_indices = torch.tensor(new_req_pool_indices, device=self.device, dtype=torch.int32)
        if new_req_pool_indices is None:
            logger.error("recv_req_migrate: Failed to allocate request pool indices (OOM in req pool). Cannot receive migrated requests.")
            # Potentially trigger some OOM handling or re-queueing for the migrated_infos
            return
        
        # b. Calculate total KV slots needed and allocate them
        total_kv_slots_needed = sum(info.kvcache_len for info in migration_infos)
        if total_kv_slots_needed > 0:
            new_physical_kv_slots_flat = self.token_to_kv_pool_allocator.alloc(total_kv_slots_needed)
            if new_physical_kv_slots_flat is None:
                logger.error("recv_req_migrate: Failed to allocate physical KV slots (OOM in KV pool). Cannot receive migrated requests.")
                self.req_to_token_pool.free(new_req_pool_indices.tolist()) # Free a.
                return
        else:
            new_physical_kv_slots_flat = torch.empty((0,), dtype=torch.int32, device=self.device)

        # --- 2. Prepare data for batch append/merge ---
        new_reqs_list: List[Req] = []
        new_seq_lens_list: List[int] = []
        
        # For EagleDraftInput (spec_info)
        new_spec_hidden_states_list: List[torch.Tensor] = []
        new_spec_topk_p_list: List[torch.Tensor] = []
        new_spec_topk_index_list: List[torch.Tensor] = []
        new_spec_verified_id_list: List[int] = []
        # Keep track of the req_pool_idx for spec_info merging
        spec_info_req_pool_indices_for_new_reqs: List[int] = []


        current_physical_slot_offset = 0
        draft_model_config = self.model_runner.model_config # Assuming draft
        target_model_config = self.target_worker.model_runner.model_config # Assuming target

        for i, mig_info in enumerate(migration_infos):
            new_req_pool_idx = new_req_pool_indices[i].item()
            
            # a. Restore KV Cache to new physical slots
            if mig_info.kvcache_len > 0:
                slots_for_this_req = new_physical_kv_slots_flat[
                    current_physical_slot_offset : current_physical_slot_offset + mig_info.kvcache_len
                ]
                
                # Write to req_to_token_pool map
                # Assuming self.req_to_token_pool.req_to_token is the map [max_total_reqs, max_seq_len_in_pool]
                # Ensure no out-of-bounds for max_seq_len_in_pool
                if mig_info.kvcache_len > self.req_to_token_pool.req_to_token.shape[1]:
                    logger.error(f"recv_req_migrate: kvcache_len {mig_info.kvcache_len} for req_pool_idx {new_req_pool_idx} "
                                 f"exceeds req_to_token_pool capacity {self.req_to_token_pool.req_to_token.shape[1]}. Skipping KV restore for this req.")
                else:
                    self.req_to_token_pool.req_to_token[new_req_pool_idx, :mig_info.kvcache_len] = slots_for_this_req

                    # Copy K and V data to physical buffers
                    # Assuming draft and target use the same KV pool for simplicity here,
                    # or that the migration info contains KV for the *current worker's* models.
                    # If they are for different types of models, this needs more complex handling.
                    # For now, let's assume mig_info.draft_model_kvcache_* is for self.model_runner
                    
                    # Draft Model KV
                    for layer_idx in range(draft_model_config.num_hidden_layers):
                        k_dest_buffer = self.model_runner.token_to_kv_pool.get_key_buffer(layer_idx)
                        v_dest_buffer = self.model_runner.token_to_kv_pool.get_value_buffer(layer_idx)
                        if layer_idx < len(mig_info.draft_model_kcache): # Check if data exists
                            k_dest_buffer.index_copy_(0, slots_for_this_req, mig_info.draft_model_kcache[layer_idx])
                            v_dest_buffer.index_copy_(0, slots_for_this_req, mig_info.draft_model_vcache[layer_idx])
                        else:
                             logger.warning(f"recv_req_migrate: Missing draft KV cache data for layer {layer_idx} for req_pool_idx {new_req_pool_idx}")
                    
                    # Target Model KV (if different physical pool or if this worker also runs target)

                    for layer_idx in range(target_model_config.num_hidden_layers):
                        k_dest_buffer_target = self.target_worker.model_runner.token_to_kv_pool.get_key_buffer(layer_idx)
                        v_dest_buffer_target = self.target_worker.model_runner.token_to_kv_pool.get_value_buffer(layer_idx)
                        if layer_idx < len(mig_info.target_model_kcache):
                            k_dest_buffer_target.index_copy_(0, slots_for_this_req, mig_info.target_model_kcache[layer_idx])
                            v_dest_buffer_target.index_copy_(0, slots_for_this_req, mig_info.target_model_vcache[layer_idx])
                        else:
                            logger.warning(f"recv_req_migrate: Missing target KV cache data for layer {layer_idx} for req_pool_idx {new_req_pool_idx}")
                
                current_physical_slot_offset += mig_info.kvcache_len

            # b. Prepare lists for batch update
            mig_info.request.req_pool_idx = new_req_pool_idx # Update req object with new pool index
            new_reqs_list.append(mig_info.request)
            new_seq_lens_list.append(mig_info.kvcache_len)

            # c. Prepare data for spec_info update
            new_spec_hidden_states_list.append(mig_info.last_hidden_state.unsqueeze(0)) # Add batch dim
            # topk_p and topk_index are List[float/int], need to convert to tensor (1, topk)
            new_spec_topk_p_list.append(torch.tensor([mig_info.topk_p], device=self.device, dtype=torch.float32))
            new_spec_topk_index_list.append(torch.tensor([mig_info.topk_index], device=self.device, dtype=torch.long))
            new_spec_verified_id_list.append(mig_info.verified_id)
            spec_info_req_pool_indices_for_new_reqs.append(new_req_pool_idx)


        # --- 3. Merge/Append to the existing batch or create new if batch is empty ---
        # a. Update basic batch attributes
        batch.reqs.extend(new_reqs_list)
        
        current_req_pool_indices = batch.req_pool_indices if batch.req_pool_indices is not None else torch.empty((0,), dtype=torch.long, device=self.device)
        batch.req_pool_indices = torch.cat((current_req_pool_indices, new_req_pool_indices))
        
        current_seq_lens = batch.seq_lens if batch.seq_lens is not None else torch.empty((0,), dtype=torch.long, device=self.device)
        batch.seq_lens = torch.cat((current_seq_lens, torch.tensor(new_seq_lens_list, device=self.device, dtype=torch.long)))
        
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()

        # b. Update batch.spec_info (EagleDraftInput)
        if new_spec_hidden_states_list: # If there's spec_info to add
            migrated_spec_info = EagleDraftInput(
                hidden_states=torch.cat(new_spec_hidden_states_list, dim=0),
                topk_p=torch.cat(new_spec_topk_p_list, dim=0),
                topk_index=torch.cat(new_spec_topk_index_list, dim=0),
                verified_id=torch.tensor(new_spec_verified_id_list, device=self.device, dtype=torch.long),
            )
            if batch.spec_info is None:
                batch.spec_info = migrated_spec_info
            else:
                batch.spec_info.merge_batch(migrated_spec_info)

        # c. Update input_ids and out_cache_loc for DECODE mode
        # Assuming these migrated requests are now ready for a DECODE step
        # input_ids for decode is usually the last verified_id of each request
        if batch.input_ids is not None and new_spec_verified_id_list:
            batch.input_ids = torch.cat((batch.input_ids, torch.tensor(new_spec_verified_id_list, device=self.device, dtype=torch.long)))
        elif new_spec_verified_id_list: # If batch.input_ids was None
            batch.input_ids = torch.tensor(new_spec_verified_id_list, device=self.device, dtype=torch.long)


        # d. Rebuild/Update SamplingInfo
        # This is crucial. SamplingInfo often depends on current batch state.
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, target_model_config.vocab_size)

        # e. Update batch mode if necessary (e.g., if it was empty, now it's DECODE)
        if batch.forward_mode is None and batch.reqs:
            batch.forward_mode = ForwardMode.DECODE # Assuming they are ready for decode

        logger.info(f"recv_req_migrate: Batch size increased to {batch.batch_size()}.")

def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int32)
