import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import parse_log, read_json_file, sem


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
    #                     help='Number of rounds of experiments.')plot_

    parser.add_argument('--model-A-profile', type=str, default=None,
                        help='Profile of model A')
    parser.add_argument('--model-B-profile', type=str, default=None,
                        help='Profile of model B')

    return parser.parse_args()


def prepare_file_paths(log_dir, prefix, sync):
    parent_folder = os.path.join(log_dir, f"{prefix}_{sync}")

    model_A_log = os.path.join(parent_folder, "model_A.csv")
    if not os.path.exists(model_A_log):
        parent_folder = os.path.join(log_dir, f"{prefix}_{sync}", "round_0")
        model_A_log = os.path.join(parent_folder, "model_A.csv")
    if not os.path.exists(model_A_log):
        raise ValueError(f"Cannot find model_A_log, {model_A_log}")

    model_B_log = os.path.join(parent_folder, "model_B.csv")
    if not os.path.exists(model_B_log):
        raise ValueError(f"Cannot find model_B_log, {model_B_log}")

    model_pid = os.path.join(parent_folder, "models_pid.json")

    return model_A_log, model_B_log, model_pid

def draw_subplot(ax, xvals, yvals, xerrs, yerrs, texts, model_A_name,
                 model_B_name, is_relative, title):
    ax.errorbar(xvals, yvals, yerr=yerrs, xerr=xerrs, fmt='o')
    for x, y, text in zip(xvals, yvals, texts):
        ax.annotate(text, (x, y))
    if is_relative:
        ax.set_xlim(0, )
        ax.set_ylim(0, )
        ax.set_xlabel(f"{model_B_name} delay inflation")
        ax.set_ylabel(f"{model_A_name} delay inflation")
    else:
        ax.set_xlabel(f"{model_B_name} avg jct (ms)")
        ax.set_ylabel(f"{model_A_name} avg jct (ms)")
    ax.set_title(title)

def filter_log(model_A_log, model_B_log):
    # filter out model_A jobs which are in collision with model_B jobs
    mask = np.ones(model_A_log.shape[0], dtype=bool)
    for _, row in model_B_log.iterrows():
        m1 = model_A_log['start_timestamp_ns'] >= row['end_timestamp_ns']
        m2 = model_A_log['end_timestamp_ns'] <= row['start_timestamp_ns']
        mask = mask & (m1 | m2)
    return model_A_log[mask]

def main():
    args = parse_args()
    if args.model_A_profile:
        model_A_profile = parse_log(args.model_A_profile)
        avg_model_A_profiled_jct = np.mean(model_A_profile['jct_ms'])
        # avg_model_A_profiled_jct = np.mean(get_jcts_from_profile(args.model_A_profile))
    else:
        avg_model_A_profiled_jct = None
    if args.model_B_profile:
        model_B_profile = parse_log(args.model_B_profile)
        avg_model_B_profiled_jct = np.mean(model_B_profile['jct_ms'])
        # avg_model_B_profiled_jct = np.mean(get_jcts_from_profile(args.model_B_profile))
    else:
        avg_model_B_profiled_jct = None
    is_relative = avg_model_A_profiled_jct is not None and \
        avg_model_B_profiled_jct is not None

    xvals, xerrs = [], []
    yvals, yerrs = [], []
    yvals_filtered, yerrs_filtered = [], []
    model_A_name, model_B_name = "", ""
    texts = []
    for sync in args.syncs:
        model_A_log_fname, model_B_log_fname, model_pid = prepare_file_paths(
            args.log_dir, args.prefix, sync)

        exp_pids = read_json_file(model_pid)
        model_A_log = parse_log(model_A_log_fname)
        model_B_log = parse_log(model_B_log_fname)
        model_A_log_filtered = filter_log(model_A_log, model_B_log)

        jcts = model_A_log['jct_ms']
        model_A_name = exp_pids[0][0]
        yvals.append(np.mean(jcts))
        yvals_filtered.append(np.mean(model_A_log_filtered['jct_ms']))
        if is_relative:
            yerrs.append(sem((jcts - avg_model_A_profiled_jct) / avg_model_A_profiled_jct))
            yerrs_filtered.append(sem((model_A_log_filtered['jct_ms'] - avg_model_A_profiled_jct) / avg_model_A_profiled_jct))
        else:
            yerrs.append(sem(jcts))
            yerrs_filtered.append(sem(model_A_log_filtered['jct_ms']))

        jcts = model_B_log['jct_ms']
        model_B_name = exp_pids[1][0]
        xvals.append(np.mean(jcts))
        if is_relative:
            xerrs.append(sem((jcts - avg_model_B_profiled_jct) / avg_model_B_profiled_jct))
        else:
            xerrs.append(sem(jcts))

        texts.append(f"sync={sync}")

    if is_relative:
        xvals = (np.array(xvals) - avg_model_B_profiled_jct) / avg_model_B_profiled_jct
        yvals = (np.array(yvals) - avg_model_A_profiled_jct) / avg_model_A_profiled_jct
        yvals_filtered = (np.array(yvals_filtered) - avg_model_A_profiled_jct) / avg_model_A_profiled_jct

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    title = ""
    draw_subplot(axes[0], xvals, yvals, xerrs, yerrs, texts,
                 model_A_name, model_B_name, is_relative, title)

    title = "Filtered jobs in collision"
    draw_subplot(axes[1], xvals, yvals_filtered, xerrs, yerrs_filtered, texts,
                 model_A_name, model_B_name, is_relative, title)

    fig.set_tight_layout(True)
    fig.savefig(os.path.join(args.log_dir, "jct_scatter_plot.jpg"))

if __name__ == '__main__':
    main()
