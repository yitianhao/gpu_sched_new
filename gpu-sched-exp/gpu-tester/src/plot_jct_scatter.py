import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import parse_log, read_json_file, get_jcts_from_profile, sem


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot scatter plot of JCT of large job vs. small job. '
        'Scatter dots are across different sync interval.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--syncs', type=int, nargs='+',
                        default=[1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 5000,
                                 10000, 1000000], help='Frequency of calling '
                        'cudaSynchronize. For example, sync = 10 means calling'
                        'cudaSynchronize every 10 kernelLauch.')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Directory containing multiple rounds of log '
                        'files and model pid file.')
    parser.add_argument('--prefix', type=str, default="sync",
                        choices=("sync", "event_group"),
                        help='Prefix of folder name.')
    # parser.add_argument('--round', type=int, default=1,
    #                     help='Number of rounds of experiments.')

    parser.add_argument('--model-A-profile', type=str, default=None,
                        help='Profile of model A')
    parser.add_argument('--model-B-profile', type=str, required=None,
                        help='Profile of model B')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_A_profile:
        avg_model_A_profiled_jct = np.mean(get_jcts_from_profile(args.model_A_profile))
    else:
        avg_model_A_profiled_jct = None
    if args.model_B_profile:
        avg_model_B_profiled_jct = np.mean(get_jcts_from_profile(args.model_B_profile))
    else:
        avg_model_B_profiled_jct = None

    xvals = []
    xerrs = []
    yvals = []
    yerrs = []
    texts = []
    for sync in args.syncs:
        parent_folder = os.path.join(args.log_dir, f"{args.prefix}_{sync}")

        model_A_log = os.path.join(parent_folder, "model_A.log")
        if not os.path.exists(model_A_log):
            parent_folder = os.path.join(args.log_dir, f"{args.prefix}_{sync}", "round_0")
            model_A_log = os.path.join(parent_folder, "model_A.log")
        if not os.path.exists(model_A_log):
            raise ValueError(f"Cannot find model_A_log, {model_A_log}")

        model_B_log = os.path.join(parent_folder, "model_B.log")
        if not os.path.exists(model_B_log):
            raise ValueError("Cannot find model_B_log, {model_B_log}")

        model_pid = os.path.join(parent_folder, "models_pid.json")
        exp_pids = read_json_file(model_pid)

        jcts = np.array(parse_log(model_A_log))
        model_A_name = exp_pids[0][0]
        yvals.append(np.mean(jcts))
        if avg_model_A_profiled_jct is not None and avg_model_B_profiled_jct is not None:
            yerrs.append(sem((jcts - avg_model_A_profiled_jct) / avg_model_A_profiled_jct))
        else:
            yerrs.append(sem(jcts))

        jcts = np.array(parse_log(model_B_log))
        model_B_name = exp_pids[1][0]
        xvals.append(np.mean(jcts))
        if avg_model_A_profiled_jct is not None and avg_model_B_profiled_jct is not None:
            xerrs.append(sem((jcts - avg_model_B_profiled_jct) / avg_model_B_profiled_jct))
        else:
            xerrs.append(sem(jcts))

        texts.append(f"sync={sync}")

    if avg_model_A_profiled_jct is not None and avg_model_B_profiled_jct is not None:
        xvals = (np.array(xvals) - avg_model_B_profiled_jct) / avg_model_B_profiled_jct
        yvals = (np.array(yvals) - avg_model_A_profiled_jct) / avg_model_A_profiled_jct

    fig, ax = plt.subplots(1, 1)

    ax.errorbar(xvals, yvals, yerr=yerrs, xerr=xerrs, fmt='o')
    for x, y, text in zip(xvals, yvals, texts):
        ax.annotate(text, (x, y))
    if avg_model_A_profiled_jct is not None and avg_model_B_profiled_jct is not None:
        ax.set_xlim(0, )
        ax.set_ylim(0, )
        ax.set_xlabel(f"{model_B_name} delay inflation")
        ax.set_ylabel(f"{model_A_name} delay inflation")
    else:
        ax.set_xlabel(f"{model_B_name} avg jct (ms)")
        ax.set_ylabel(f"{model_A_name} avg jct (ms)")

    fig.set_tight_layout(True)
    fig.savefig(os.path.join(args.log_dir, "jct_scatter_plot.jpg"))

if __name__ == '__main__':
    main()
