import pickle
from typing import Any, Tuple

import zmq
from loguru import logger


class ZmqServer:
    def __init__(self, port: int):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt_string(zmq.IDENTITY, "hub")
        self.socket.bind(f"tcp://*:{self.port}")
        logger.info(f"ZmqServer started on tcp://*:{self.port}")

    def send(self, identity: bytes, message: Any):
        message = pickle.dumps(message)
        self.socket.send_multipart([identity, message])

    def recv(self) -> Tuple[bytes, Any]:
        identity, message = self.socket.recv_multipart()
        return identity, pickle.loads(message)

    def has_data(self) -> bool:
        return self.socket.poll(0) == zmq.POLLIN


class ZmqClient:
    def __init__(self, client_id: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.client_id = client_id

    def connect(self, address):
        self.socket.setsockopt_string(zmq.IDENTITY, self.client_id, encoding="utf-8")
        self.socket.connect(address)

    def send(self, message: Any):
        message = pickle.dumps(message)
        self.socket.send(message)

    def recv(self) -> Any:
        message = self.socket.recv()
        return pickle.loads(message)

    def has_data(self) -> bool:
        return self.socket.poll(0) == zmq.POLLIN
