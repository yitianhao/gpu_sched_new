import numpy as np
from typing import Optional
from exp_log import GPUJobProfile


SEC_IN_NS = 1e9
SEC_IN_MS = 1e3

class Predictor:
    def __init__(self, model_A_prof, model_B_prof:Optional[GPUJobProfile] = None):
        self.model_A_prof = model_A_prof
        self.model_B_prof = model_B_prof

    def predict_optimal_sync_freq(self, sync_freqs=[1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 5000, 10000, 1000000]):
        delay_sum_list = []
        for sync in sync_freqs:
            exe_delay = self.predict_execution_delay(sync)
            preemption_delay = self.predict_preemption_delay(sync)
            delay_sum_list.append(exe_delay + preemption_delay)
        idx = np.argmin(delay_sum_list)
        return sync_freqs[idx]

    def predict_preemption_delay(self, sync_freq):
        """Predict preemption delay in ms."""
        return min(sync_freq, max(self.model_A_prof.num_kernels_queued)) * \
                self.model_A_prof.get_kernel_exec_time_mean() / SEC_IN_NS * SEC_IN_MS

    def predict_execution_delay(self, sync_freq, a=0.0115, b=2.92):
        """Predict execution delay in ms."""
        n_sync_calls = len(self.model_A_prof) // sync_freq
        return a * n_sync_calls + b # TODO: verify correctness
