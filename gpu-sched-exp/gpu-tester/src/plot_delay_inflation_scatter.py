import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import parse_log, read_json_file, get_jcts_from_profile, sem


COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
          'olive', 'cyan', 'black', 'navy', 'yellow']
MARKERS = ['o', '+', 'x', '*', '.', 'X']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot scatter plot of JCT of large job vs. small job. '
        'Scatter dots are across different sync interval and across different '
        'job pairs.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--syncs', type=int, nargs='+',
                        default=[1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 5000,
                                 10000, 1000000], help='Frequency of calling '
                        'cudaSynchronize. For example, sync = 10 means calling'
                        'cudaSynchronize every 10 kernelLauch.')
    parser.add_argument('--log-dirs', type=str, nargs='+', required=True,
                        help='Directory containing multiple rounds of log '
                        'files and model pid file.')
    parser.add_argument('--prefix', type=str, default='sync', choices=("sync", "event_group"),
                        help='Prefix of folder name.')
    # parser.add_argument('--round', type=int, default=1,
    #                     help='Number of rounds of experiments.')

    parser.add_argument('--model-A-profiles', type=str, nargs='+', required=True,
                        help='Profiles of model A\'s')
    parser.add_argument('--model-B-profiles', type=str, nargs='+', required=True,
                        help='Profiles of model B\'s')

    return parser.parse_args()


def prepare_file_paths(log_dir, prefix, sync):
    parent_folder = os.path.join(log_dir, f"{prefix}_{sync}")

    model_A_log = os.path.join(parent_folder, "model_A.log")
    if not os.path.exists(model_A_log):
        parent_folder = os.path.join(log_dir, f"{prefix}_{sync}", "round_0")
        model_A_log = os.path.join(parent_folder, "model_A.log")
    if not os.path.exists(model_A_log):
        raise ValueError(f"Cannot find model_A_log, {model_A_log}")

    model_B_log = os.path.join(parent_folder, "model_B.log")
    if not os.path.exists(model_B_log):
        raise ValueError("Cannot find model_B_log, {model_B_log}")

    model_pid = os.path.join(parent_folder, "models_pid.json")

    return model_A_log, model_B_log, model_pid

def main():
    args = parse_args()
    assert len(args.log_dirs) == len(args.model_A_profiles)
    assert len(args.log_dirs) == len(args.model_B_profiles)
    avg_model_A_profiled_jcts = [np.mean(get_jcts_from_profile(profile))
                                 for profile in args.model_A_profiles]
    avg_model_B_profiled_jcts = [np.mean(get_jcts_from_profile(profile))
                                 for profile in args.model_B_profiles]


    color_legend_patches = []
    fig, ax = plt.subplots(1, 1)
    for avg_model_A_profiled_jct, avg_model_B_profiled_jct, log_dir, mkr in zip(
        avg_model_A_profiled_jcts, avg_model_B_profiled_jcts, args.log_dirs, MARKERS):
        print(log_dir)
        model_name = os.path.splitext(os.path.basename(log_dir))[0]
        model_A_inf = []
        model_B_inf = []
        xerrs = []
        yerrs = []
        texts = []

        for sync in args.syncs:
            try:
                model_A_log, model_B_log, model_pid = prepare_file_paths(
                    log_dir, args.prefix, sync)
            except ValueError:
                continue
            exp_pids = read_json_file(model_pid)

            jcts = parse_log(model_A_log)
            model_A_name = exp_pids[0][0]
            # model_A_avg_jcts.append(np.mean(jcts))
            model_A_inf.append((np.mean(jcts) - avg_model_A_profiled_jct) / avg_model_A_profiled_jct)

            jcts = parse_log(model_B_log)
            model_B_name = exp_pids[1][0]
            # model_B_avg_jcts.append(np.mean(jcts))
            model_B_inf.append((np.mean(jcts) - avg_model_B_profiled_jct) / avg_model_B_profiled_jct)

            # texts.append(f"sync={sync}")
        colors = COLORS[:len(model_A_inf)]

        dots = ax.scatter(model_B_inf, model_A_inf, marker=mkr, c=colors, label=model_name)
        color_legend_patches.append(dots)


    colors = COLORS[:len(args.syncs)]
    for c, sync in zip(colors, args.syncs):
        color_legend_patches.append(mpatches.Patch(color=c, label=f'sync={sync}'))

    ax.legend(handles=color_legend_patches)

    ax.set_xlim(0, )
    ax.set_ylim(0, )
    # ax.legend()
    ax.set_xlabel("Small job delay inflation")
    ax.set_ylabel("Large job delay inflation")
    # ax.errorbar(xvals, yvals, yerr=yerrs, xerr=xerrs, fmt='o')
    # for x, y, text in zip(xvals, yvals, texts):
    #     ax.annotate(text, (x, y))
    # if avg_model_A_profiled_jct is not None and avg_model_B_profiled_jct is not None:
    # else:
    #     ax.set_xlabel(f"{model_B_name} avg jct (ms)")
    #     ax.set_ylabel(f"{model_A_name} avg jct (ms)")

    fig.set_tight_layout(True)
    fig.savefig("jct_scatter_plot_across_model_pairs.jpg")


if __name__ == '__main__':
    main()
