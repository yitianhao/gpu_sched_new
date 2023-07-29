import argparse
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import parse_log

SEC_IN_NS = 1e9
SEC_IN_MS = 1e3

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot job completion time vs. time for two jobs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-A-jct-log', type=str, required=True,
                        help='Model A jct log (.csv).')
    parser.add_argument('--model-B-jct-log', type=str, required=True,
                        help='Model B jct log (.csv).')
    parser.add_argument('--save-dir', type=str, default='.',
                        help='image save dir.')

    return parser.parse_args()


def plot_jct_vs_time(model_A_log, model_B_log, model_A_name, model_B_name, save_dir):
    min_t = min(model_A_log['start_timestamp_ns'].iloc[0],
                model_B_log['start_timestamp_ns'].iloc[0])
    max_t = max(model_A_log['end_timestamp_ns'].iloc[-1],
                model_B_log['end_timestamp_ns'].iloc[-1])
    fig = plt.figure(figsize=(16, 8))
    ax = plt.gca()

    ax.plot((model_A_log['end_timestamp_ns'] - min_t) / SEC_IN_NS,
            model_A_log['jct_ms'], 'o-', label=model_A_name)
    ax.plot((model_B_log['end_timestamp_ns'] - min_t) / SEC_IN_NS,
            model_B_log['jct_ms'], 'o-', label=model_B_name)
    ax.set_ylabel("JCT (ms)")
    ax.set_xlim(0, (max_t - min_t) / SEC_IN_NS + 0.1)
    ax.set_xlabel('Time (s)')
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(os.path.join(save_dir, 'jct_vs_t.jpg'), bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    model_A_jct_log = parse_log(args.model_A_jct_log)
    model_B_jct_log = parse_log(args.model_B_jct_log)

    plot_jct_vs_time(model_A_jct_log, model_B_jct_log, 'model_A job',
                     'model_B job', args.save_dir)


if __name__ == '__main__':
    main()
