import os
from typing import Optional
import pandas as pd
import numpy as np
from utils import read_json_file, write_json_file


class GPUSchedExpLog:
    def __init__(self, nsys_kernel_trace: str, nsys_nvtx_trace: str,
                 model_A_jct_log: str, model_B_jct_log: str, pid_log: str):
        self.kernel_trace = pd.read_csv(nsys_kernel_trace)
        self.kernel_trace['Kernel End (ns)'] = \
            self.kernel_trace['Kernel Start (ns)'] + self.kernel_trace['Kernel Dur (ns)']
        self.nvtx_trace = pd.read_csv(nsys_nvtx_trace)

        self.model_A_jct_log = pd.read_csv(model_A_jct_log)
        self.model_B_jct_log = pd.read_csv(model_B_jct_log)

        self.models_pid = read_json_file(pid_log)
        # get model A GPU kernels
        mask = self.kernel_trace['PID'] == self.models_pid[0][1]
        self.model_A_kernel_trace = self.kernel_trace[mask]

        # get model B GPU kernels
        mask = self.kernel_trace['PID'] == self.models_pid[1][1]
        self.model_B_kernel_trace = self.kernel_trace[mask]

        self.model_A_job_num_kernels = len(self.model_A_kernel_trace.index) / len(self.model_A_jct_log.index)
        self.model_B_job_num_kernels = len(self.model_B_kernel_trace.index) / len(self.model_B_jct_log.index)
        if self.model_A_job_num_kernels.is_integer():
            self.model_A_job_num_kernels = int(self.model_A_job_num_kernels)
        else:
            raise ValueError(f"model A job num kernels is {self.model_A_job_num_kernels}")
        if self.model_B_job_num_kernels.is_integer():
            self.model_B_job_num_kernels = int(self.model_B_job_num_kernels)
        else:
            raise ValueError(f"model B job num kernels is {self.model_B_job_num_kernels}")
        print(self.model_A_job_num_kernels, self.model_B_job_num_kernels)

    def get_model_B_jct_ms(self):
        return self.model_B_jct_log['jct_ms'].tolist()

    def get_kernel_queue(self):
        """Get a list of kernel ids of model A GPU kernels in queue when model
        B arrives. Each kernel id is in [0, num_kernels in one model A job]."""

        kernel_ids_groups = []

        # get model B job start and end time
        mask = (self.nvtx_trace['PID'] == self.models_pid[1][1]) & \
               (self.nvtx_trace['Name'] == "regionTest")
        for _, row in self.nvtx_trace[mask].iterrows():
            # get model B GPU kernels with in a model B job
            m1 = ((row['Start (ns)'] <= self.model_B_kernel_trace['Kernel Start (ns)'])
            & (self.model_B_kernel_trace['Kernel Start (ns)'] <= row['End (ns)']))
            m2 = ((row['Start (ns)'] <= self.model_B_kernel_trace['Kernel End (ns)']) &
            (self.model_B_kernel_trace['Kernel End (ns)'] <= row['End (ns)']))
            model_B_kernels = self.model_B_kernel_trace[m1 | m2]

            # model B's kernels' timestamps within regionTest range
            model_B_start = model_B_kernels.iloc[0]['Kernel Start (ns)']
            model_B_end = model_B_kernels.iloc[-1]['Kernel End (ns)']

            # model B's regionTest start and end timestamps
            # model_B_start = row['Start (ns)']
            # model_B_end = row['End (ns)']

            # get model A GPU kernels which have overlaps with model B job
            m1 = ((model_B_start <= self.model_A_kernel_trace['Kernel Start (ns)'])
            & (self.model_A_kernel_trace['Kernel Start (ns)'] <= model_B_end))
            m2 = ((model_B_start <= self.model_A_kernel_trace['Kernel End (ns)']) &
            (self.model_A_kernel_trace['Kernel End (ns)'] <= model_B_end))

            kernel_ids = []
            for idx, row in self.model_A_kernel_trace[m1 | m2].iterrows():
                kernel_ids.append(self.model_A_kernel_trace.index.get_loc(idx)
                                  % self.model_A_job_num_kernels)
            kernel_ids_groups.append(kernel_ids)
        return kernel_ids_groups


class GPUJobProfile:
    def __init__(self, nsys_kernel_profile: str, jct_profile: str,
                 nsys_nvtx_profile: Optional[str] = None,
                 num_kernels: Optional[int] = None,
                 cache: bool = True, overwrite=True):
        self.nsys_kernel_profile = nsys_kernel_profile
        self.nsys_nvtx_profile = nsys_nvtx_profile
        self.jct_profile = jct_profile
        folder = os.path.dirname(nsys_kernel_profile)
        cache_fname = os.path.join(folder, "kernel_exec_time_map.json")
        self.jct_profile = pd.read_csv(jct_profile)
        if not overwrite and cache and os.path.exists(cache_fname):
            # speed up profile log reading
            self.mean_kernel_exec_time_map = read_json_file(cache_fname)
            self.num_kernels = len(self.mean_kernel_exec_time_map)
            return
        self.kernel_profile = pd.read_csv(nsys_kernel_profile)

        if num_kernels is None:
            self.num_kernels = len(self.kernel_profile.index) / len(self.jct_profile.index)
            if self.num_kernels.is_integer():
                self.num_kernels = int(self.num_kernels)
            else:
                raise ValueError(f"num kernels is {self.num_kernels}")
        else:
            self.num_kernels = num_kernels
        self._build_kernel_execution_time_map()
        self._build_mean_kernel_execution_time_map()
        if overwrite or (cache and not os.path.exists(cache_fname)):
            write_json_file(cache_fname, self.mean_kernel_exec_time_map)

        if nsys_nvtx_profile is not None:
            self.nvtx_profile = pd.read_csv(nsys_nvtx_profile)

    def _build_kernel_execution_time_map(self):
        self.kernel_exec_time_map = dict()  # map id to execution time in ns
        for i, (_, row) in enumerate(self.kernel_profile.iterrows()):
            kernel_idx = i % self.num_kernels
            if kernel_idx not in self.kernel_exec_time_map:
                self.kernel_exec_time_map[kernel_idx] = (row['Kernel Name'], [])
            assert row['Kernel Name'] == self.kernel_exec_time_map[kernel_idx][0]
            self.kernel_exec_time_map[kernel_idx][1].append(row['Kernel Dur (ns)'])

    def _build_mean_kernel_execution_time_map(self):
        self.mean_kernel_exec_time_map = dict()
        for idx, (name, durs) in self.kernel_exec_time_map.items():
            self.mean_kernel_exec_time_map[idx] = {
                'kernel name': name, 'mean': np.mean(durs),
                'std': np.std(durs), 'count': len(durs)}

    def __getitem__(self, kernel_index):
        return self.mean_kernel_exec_time_map[kernel_index]

    def __len__(self):
        return len(self.mean_kernel_exec_time_map)

    def get_avg_jct_ms(self):
        return self.jct_profile.iloc[1:]['jct_ms'].mean()

    def get_kernel_exec_times(self, kernel_ids):
        return [self.mean_kernel_exec_time_map[id]['mean'] for id in kernel_ids]
