import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.collections as mc
import matplotlib.pyplot as plt
from utils import parse_log, read_json_file, sem

SEC_IN_NS = 1e9
SEC_IN_MS = 1e3

COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
          'olive', 'cyan', 'black', 'navy', 'yellow']

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
    ax.errorbar(xvals, yvals, yerr=yerrs, xerr=xerrs, color='none', ecolor=COLORS[:len(xvals)])
    ax.scatter(xvals, yvals, marker='o', color=COLORS[:len(xvals)])
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

def plot_job_arrival(model_A_log, model_B_log, model_A_name, model_B_name, save_dir):
    model_A_color = 'C0'
    model_B_color = 'C3'
    fig = plt.figure(figsize=(24, 8))
    ax = plt.gca()
    lines = []
    for start, end in zip(
        model_A_log['start_timestamp_ns'], model_A_log['end_timestamp_ns']):
        lines.append([(start / SEC_IN_NS, 1), (end / SEC_IN_NS, 1)])
    lc = mc.LineCollection(lines, colors=model_A_color, linewidths=4)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_ylabel("Job arrival\nof {}".format(model_A_name), color=model_A_color)
    ax.vlines(x=model_A_log['end_timestamp_ns'] / SEC_IN_NS, ymin=0, ymax = 1, colors='k', ls='--')

    lines = []
    for start, end in zip(model_B_log['start_timestamp_ns'], model_B_log['end_timestamp_ns']):
        lines.append([(start / SEC_IN_NS, 0.5), (end / SEC_IN_NS, 0.5)])
    lc = mc.LineCollection(lines, colors=model_B_color, linewidths=4)
    ax = ax.twinx()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_ylim(0, 1.1)
    # ax.set_xlim(0, t_max)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Job arrival\nof {}".format(model_B_name), color=model_B_color)
    ax.set_yticks([])
    fig.set_tight_layout(True)
    fig.savefig(os.path.join(save_dir, 'job_arrival.jpg'), bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    if args.model_A_profile:
        model_A_profile = parse_log(args.model_A_profile)
        avg_model_A_prof_jct = np.mean(model_A_profile['jct_ms'])
        # avg_model_A_prof_jct = np.mean(get_jcts_from_profile(args.model_A_profile))
    else:
        avg_model_A_prof_jct = None
    if args.model_B_profile:
        model_B_profile = parse_log(args.model_B_profile)
        avg_model_B_prof_jct = np.mean(model_B_profile['jct_ms'])
        # avg_model_B_prof_jct = np.mean(get_jcts_from_profile(args.model_B_profile))
    else:
        avg_model_B_prof_jct = None
    is_relative = avg_model_A_prof_jct is not None and \
        avg_model_B_prof_jct is not None

    xvals, xerrs, yvals, yerrs  = [], [], [], []
    xvals_abs, xerrs_abs, yvals_abs, yerrs_abs = [], [], [], []
    model_A_name, model_B_name = "", ""
    texts = []
    syncs = []
    for sync in args.syncs:
        try:
            model_A_log_fname, model_B_log_fname, model_pid = prepare_file_paths(
                args.log_dir, args.prefix, sync)
        except ValueError:
            continue

        exp_pids = read_json_file(model_pid)
        model_A_log = parse_log(model_A_log_fname)
        model_B_log = parse_log(model_B_log_fname)
        exp_config = read_json_file(os.path.join(os.path.dirname(model_A_log_fname), 'exp_config.json'))
        model_A_batch_size = exp_config['models'][0]['batch_size']
        model_A_name = exp_pids[0][0] + '_batch_size_' + str(model_A_batch_size)
        model_B_name = exp_pids[1][0]
        plot_job_arrival(model_A_log, model_B_log, model_A_name, model_B_name,
                         os.path.dirname(model_A_log_fname))
        model_A_log_filtered = filter_log(model_A_log, model_B_log)
        suptitle = "Including jobs in collision"
        if len(model_A_log_filtered) > 0:
            suptitle = "Filtered jobs in collision"
            model_A_log = model_A_log_filtered
        else:
            print(f'Skip {model_A_log_fname}, no filtered model_A logs!')

        syncs.append(sync)
        jcts = model_A_log['jct_ms']
        yvals_abs.append(np.mean(jcts))
        yerrs_abs.append(sem(jcts))
        if is_relative:
            yvals.append((np.mean(jcts) - avg_model_A_prof_jct) / avg_model_A_prof_jct)
            yerrs.append(sem((jcts - avg_model_A_prof_jct) / avg_model_A_prof_jct))
        else:
            yvals.append(np.mean(jcts))
            yerrs.append(sem(jcts))

        jcts = model_B_log['jct_ms']
        xvals_abs.append(np.mean(jcts))
        xerrs_abs.append(sem(jcts))
        if is_relative:
            xvals.append((np.mean(jcts) - avg_model_B_prof_jct) / avg_model_B_prof_jct)
            xerrs.append(sem((jcts - avg_model_B_prof_jct) / avg_model_B_prof_jct))
        else:
            xvals.append(np.mean(jcts))
            xerrs.append(sem(jcts))

        texts.append(f"sync={sync}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    title = ""
    draw_subplot(axes[0], xvals_abs, yvals_abs, xerrs_abs, yerrs_abs, texts,
                 model_A_name, model_B_name, False, title)
    # axes[0].plot(44.7, 110.1, "o")
    draw_subplot(axes[1], xvals, yvals, xerrs, yerrs, texts,
                 model_A_name, model_B_name, is_relative, title)
    if is_relative:
        axes[1].axvline(x = 0.1, c='k', ls='--')#, ymin=0, ymax=0.1)
        axes[1].axhline(y = 0.1, c='k', ls='--')#, xmin=0, xmax=0.1)
        # axes[1].set_xlim(0, 1)
        # axes[1].set_ylim(0, 1)

    # fig.suptitle(suptitle)

    fig.set_tight_layout(True)
    fig.savefig(os.path.join(args.log_dir, "jct_scatter_plot.jpg"), bbox_inches='tight')

    with open(os.path.join(args.log_dir, "jct_scatter_plot.csv"), 'w') as f:
        hd = csv.writer(f, lineterminator='\n')
        hd.writerow(["sync", "xvals_abs", "yvals_abs", "xerrs_abs", "yerrs_abs", "xvals", "yvals", "xerrs", "yerrs"])
        hd.writerows(zip(syncs, xvals_abs, yvals_abs, xerrs_abs, yerrs_abs, xvals, yvals, xerrs, yerrs))

if __name__ == '__main__':
    main()
