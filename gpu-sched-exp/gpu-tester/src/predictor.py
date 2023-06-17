import os

from exp_log import GPUJobProfile


class Predictor:
    def __init__(self, model_A_prof_root, model_B_prof_root):
        kernel_tr = os.path.join(model_A_prof_root, "nsight_report_kernexectrace.csv")
        # nvtx_tr = os.path.join(prof_root, "nsight_report_nvtxpptrace.csv")
        nvtx_tr = None
        jct_log = os.path.join(model_A_prof_root, os.path.basename(model_A_prof_root)+".csv")
        self.model_A_prof = GPUJobProfile(kernel_tr, jct_log, nvtx_tr, cache=False)

        kernel_tr = os.path.join(model_B_prof_root, "nsight_report_kernexectrace.csv")
        # nvtx_tr = os.path.join(prof_root, "nsight_report_nvtxpptrace.csv")
        nvtx_tr = None
        jct_log = os.path.join(model_B_prof_root, os.path.basename(model_B_prof_root)+".csv")
        self.model_B_prof = GPUJobProfile(kernel_tr, jct_log, nvtx_tr, cache=False)

    def predict_optimal_sync_freq(self, sync_freqs=[1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 5000, 10000, 1000000]):
        delay_sum_list = []
        for sync in sync_freqs:
            exe_delay = self.predict_execution_delay(sync)
            preemption_delay = self.predict_preemption_delay()
            delay_sum_list.append(exe_delay + preemption_delay)
        idx = np.argmin(delay_sum_list)
        return sync_freqs[idx]

    def predict_preemption_delay(self, sync_freq):
        """Predict preemption delay in ms."""
        return sync_freq * self.model_A_prof.get_avg_kernel_exec_time()

    def predict_execution_delay(self, sync_freq):
        """Predict execution delay in ms."""
        n_sync_calls = len(self.model_A_prof) / sync_freq
        return = 0.0115 * nsync_calls + 2.92 # TODO: verify correctness
