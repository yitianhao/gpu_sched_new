import multiprocessing as mp
import struct
import sys

from tcp_utils import TcpAgent, TcpServer, timestamp
from worker import WorkerProc


def main():
    # create a pipe to deliver the job request to the worker
    pipe_parent, pipe_child = mp.Pipe()

    # spawn a worker
    worker_proc = WorkerProc(pipe_child)
    worker_proc.start()

    # Accept connections
    server = TcpServer('localhost', 12345)
    timestamp('tcp', 'listen')

    # only allows one TCP connection for now
    conn, _ = server.accept()
    agent = TcpAgent(conn)
    timestamp('tcp', 'connected')

    while True:
        timestamp('tcp', 'listening')
        request_len_b = agent.recv(4)
        if not request_len_b:
            break
        request_len = struct.unpack('I', request_len_b)[0]

        request = agent.recv(request_len)
        if not request:
            break
        # print(request, file=sys.stderr, flush=True)

        # timestamp('tcp requst', request)
        pipe_parent.send(request)
        print("send request to worker", file=sys.stderr, flush=True)

        # timestamp('tcp', 'enqueue_request')
    # worker_proc.terminate()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
