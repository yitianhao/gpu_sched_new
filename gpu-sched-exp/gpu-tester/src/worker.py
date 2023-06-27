import csv
import json
import multiprocessing as mp
import os
import sys
from time import perf_counter_ns
from ctypes import cdll

import torch

from tcp_utils import timestamp
from transformer_model import is_transformer
from vision_model import is_vision_model, VisionModel


def debug_print(msg):
    print(msg, file=sys.stderr, flush=True)

class WorkerProc(mp.Process):
    def __init__(self, pipe):
        super(WorkerProc, self).__init__()
        self.pipe = pipe

        # variables used in DNN inference
        self.model = None
        self.lib = None  # so lib used to modify shared memory
        self.device_id = None
        self.csv_fname = None
        self.csv_fh = None
        self.csv_writer = None
        timestamp('tcp server worker', 'init')

    def run(self):
        # set the directory for downloading models
        torch.hub.set_dir("../torch_cache/")
        while True:
            print("in loop")
            request = self.pipe.recv()
            if not request:
                break
            request = json.loads(request.decode('utf-8'))
            # print('worker', request, file=sys.stderr, flush=True)

            # handle job request
            # initialize DNN model
            if self.device_id is None or self.device_id != request['device_id']:
                self.device_id = request['device_id']
                # set the cuda device to use
                torch.cuda.set_device(self.device_id)

            if self.model is None and is_vision_model(
                request["model_name"], request["model_weight"]):
                self.model = VisionModel(request, request["device_id"])
            elif self.model is None and is_transformer("", ""):
                raise NotImplementedError
            else:
                raise NotImplementedError

            # set up log
            os.makedirs(request['output_file_path'], exist_ok=True)
            csv_fname = os.path.join(
                request['output_file_path'],
                request['output_file_name'] + ".csv")
            if self.csv_fname is None or self.csv_fname != csv_fname:
                self.csv_fname = csv_fname
                if self.csv_fh is not None:
                    self.csv_fh.flush()
                    self.csv_fh.close()
                self.csv_fh = open(self.csv_fname, 'w', 1)
                self.csv_writer = csv.writer(self.csv_fh, lineterminator='\n')
                self.csv_writer.writerow(
                    ['start_timestamp_ns', 'end_timestamp_ns', 'jct_ms',
                     'max_allocated_gpu_memory_allocated_byte',
                     'max_reserved_gpu_memory_byte'])

            # set up so lib
            if request['control'] and request['priority'] > 0:
                # only load library when needed
                self.lib = cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so"))
            else:
                self.lib = None

            res = self.infer()
            print(res)

    def __del__(self):
        if self.csv_fh is not None:
            self.csv_fh.flush()
            self.csv_fh.close()

    def infer(self):
        start_t: int = perf_counter_ns()
        res = None # res :torch.Tensor
        if self.lib is not None:
            try:
                suffix = os.getenv("SUFFIX", None)
                assert suffix is not None
                self.lib.setMem(1, suffix.encode())
                self.lib.waitForEmptyGPU()
            except Exception as e:
                debug_print(e)
        assert self.model is not None
        res = self.model()
        torch.cuda.synchronize()
        end_t: int = perf_counter_ns()
        if self.lib is not None:
            try:
                suffix = os.getenv("SUFFIX", None)
                assert suffix is not None
                self.lib.setMem(0, suffix.encode())
            except Exception as e:
                debug_print(e)
        max_alloc_mem_byte = torch.cuda.max_memory_allocated(self.device_id)
        max_rsrv_mem_byte = torch.cuda.max_memory_reserved(self.device_id)
        if self.csv_writer is not None:
            self.csv_writer.writerow([
                start_t, end_t, (end_t - start_t) / 1000000, max_alloc_mem_byte,
                max_rsrv_mem_byte])
        return res
