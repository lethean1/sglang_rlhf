import sglang as sgl
import json
import time
import os
import torch
import torch.distributed as dist
import ray
from zmq_link import ZmqServer
import traceback

@ray.remote(num_gpus=1)
def run_on_gpu(rank, world_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    with open("/workspace/prompt.json",'r') as file:
        data = json.load(file)

    prompts = []
    idx = 0
    for d in data:
        if idx < 800:
            prompts.append(d)
            idx += 1
        else:
            break
    print(f"process: {rank},prompts len: ", len(prompts))
    
    llm = sgl.Engine(
        model_path="/data/model_zoo/llama2/llama-2-7b-chat/",
        #model_path="/dev/shm/llama2-7b-chat/",
        world_rank=rank,
        world_size=world_size,
        max_running_requests=8,
        tp_size=1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
    )
    sampling_params = {"temperature": 0, "max_new_tokens": 30}
    print(f"process {rank}, start generating: ", time.time())
    llm.generate(prompts, sampling_params)
    #time.sleep(10)
    print("generation down")
    llm.shutdown()

if __name__ == "__main__":
    ray.init()
    
    world_size = 2
    futures = []
    for rank in range(world_size):
        future = run_on_gpu.remote(rank, world_size)
        futures.append(future)
    
    identities = {}
    zmq_server = ZmqServer(8899)
    while True:
        try:
            identity, world_rank = zmq_server.recv()
            print(f"server recv msg: {world_rank}")
            identities[world_rank] = identity
            if len(identities) == world_size:
                print("all identities received")
                break
        except Exception as e:
            traceback.print_exc()
            raise e
    migration_num = 4
    migration_dst = 1
    migration_src = 0
    time.sleep(20)
    zmq_server.send(identities[migration_src], str(migration_num)+'_'+str(migration_dst)+'_'+'send')
    zmq_server.send(identities[migration_dst], str(migration_num)+'_'+str(migration_src)+'_'+'recv')
    # zmq_server.send(identities[1], "runrunrun")

    ray.get(futures)
    
    ray.shutdown()


