import argparse
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import pandas as pd
from utils import read_json_file

SEC_IN_NS = 1e9
SEC_IN_MS = 1e3

def cdf(data):
    length = len(data)
    x = np.sort(data)
    # Get the CDF values of y
    y = np.arange(length) / float(length)
    return x, y

def compute_queue(start, end):
    start = sorted(start)
    end = sorted(end)
    ts = []
    num_kernel_queued = 0
    num_kernels_queued = []
    i = 0
    j = 0

    while i < len(start) and j < len(end):
        if start[i] < end[j]:
            ts.append(start[i])
            num_kernels_queued.append(num_kernel_queued)
            num_kernel_queued += 1
            i += 1
        else:
            ts.append(end[j])
            num_kernel_queued -= 1
            num_kernels_queued.append(num_kernel_queued)
            j += 1
    return ts, num_kernels_queued


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', '--file', metavar="FILEPATH", type=str,
                        help="Path to a nsys report csv file.", required=True)
    parser.add_argument('-p', '--pids', metavar="FILEPATH",  default=None,
                        help="Specifies the path to the pid file.")
    parser.add_argument('-o', '--output-dir', metavar="OUTPUT_DIR", type=str,
                        help="Path to save the output figures.", default='.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    # csv header definition:
    # /opt/nvidia/nsight-systems/2022.4.2/host-linux-x64/reports/kernexectrace.py: Line 22
    df = pd.read_csv(args.file)

    # remove all 'None' values in Queue Start (ns)  and Queue Dur (ns) column
    df['Queue Start (ns)'].replace('None', np.nan, inplace=True)
    df['Queue Start (ns)'] = df['Queue Start (ns)'].astype(float)

    df['Queue Dur (ns)'].replace('None', np.nan, inplace=True)
    df['Queue Dur (ns)'] = df['Queue Dur (ns)'].astype(float)

    df['API End (ns)'] = df['API Start (ns)'] + df['API Dur (ns)']
    df['Kernel End (ns)'] = df['Kernel Start (ns)'] + df['Kernel Dur (ns)']
    df['End (ns)'] = df['API Start (ns)'] + df['Total Dur (ns)']

    t_min = 0
    t_max = df['End (ns)'].iloc[-1] / SEC_IN_NS

    if args.pids is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        start = df['API Start (ns)'].tolist()
        end = df['Kernel End (ns)'].tolist()
        ts, num_kernels_queued = compute_queue(start, end)

        ax = axes[0]
        ax.step(np.array(ts) / SEC_IN_NS, num_kernels_queued)
        ax.set_ylabel('Num of kernels in queue')
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, )

        ax = axes[1]
        ax.plot(df['Queue Start (ns)'] / SEC_IN_NS,
                df['Queue Dur (ns)'] / SEC_IN_NS * SEC_IN_MS)
        ax.set_ylabel('Queue Dur (ms)')
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, )

        fig.set_tight_layout(True)
        fig.savefig(os.path.join(args.output_dir, 'queued_kernels.jpg'),
                    bbox_inches='tight')
        return

    experiment_pids = read_json_file(args.pids)
    model_A_name, model_A_pid = experiment_pids[0]
    model_B_name, model_B_pid = experiment_pids[1]
    model_A_color = 'C0'
    model_B_color = 'C3'

    fig, axes = plt.subplots(6, 1, figsize=(24, 18))

    ax = axes[0]
    start = df['API Start (ns)'].tolist()
    end = df['Kernel End (ns)'].tolist()
    ts, num_kernels_queued = compute_queue(start, end)

    ax.step(np.array(ts) / SEC_IN_NS, num_kernels_queued, color='k')
    ax.set_ylabel('Num of kernels in queue')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, )

    ax = axes[1]
    mask = df['PID'] == model_A_pid

    start = df[mask]['API Start (ns)'].tolist()
    end = df[mask]['Kernel End (ns)'].tolist()
    ts, num_kernels_queued = compute_queue(start, end)

    ax.step(np.array(ts) / SEC_IN_NS, num_kernels_queued, color=model_A_color)
    ax.set_ylabel(f'Num of queued kernels\n{model_A_name}')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, )

    ax = axes[2]
    ax.scatter(df[mask]['Queue Start (ns)'] / SEC_IN_NS,
               df[mask]['Queue Dur (ns)'] / SEC_IN_NS * SEC_IN_MS,
               c=model_A_color, marker='.')
    ax.set_ylabel(f'Queue Delay (ms)\nof {model_A_name}')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, )

    ax = axes[3]
    mask = df['PID'] == model_B_pid

    start = df[mask]['API Start (ns)'].tolist()
    end = df[mask]['Kernel End (ns)'].tolist()
    ts, num_kernels_queued = compute_queue(start, end)

    ax.step(np.array(ts) / SEC_IN_NS, num_kernels_queued, color=model_B_color)
    ax.set_ylabel(f'Num of queued kernels \n{model_B_name}')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, )

    ax = axes[4]
    ax.scatter(df[mask]['Queue Start (ns)'] / SEC_IN_NS,
               df[mask]['Queue Dur (ns)'] / SEC_IN_NS * SEC_IN_MS,
               c=model_B_color, marker='.')
    ax.set_ylabel(f'Queue Delay (ms)\nof {model_B_name}')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, )

    ax = axes[5]
    mask = df['PID'] == model_B_pid
    lines = []
    for start, end in zip(df[mask]['API Start (ns)'], df[mask]['End (ns)']):
        lines.append([(start / SEC_IN_NS, 0.5), (end / SEC_IN_NS, 0.5)])

    lc = mc.LineCollection(lines, colors=model_B_color, linewidths=4)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Job arrival\nof {}".format(model_B_name), color=model_B_color)
    ax.set_yticks([])

    ax = ax.twinx()
    mask = df['PID'] == model_A_pid
    lines = []
    for start, end in zip(df[mask]['API Start (ns)'], df[mask]['End (ns)']):
        lines.append([(start / SEC_IN_NS, 1), (end/SEC_IN_NS, 1)])

    lc = mc.LineCollection(lines, colors=model_A_color, linewidths=4)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_ylabel("Job arrival\nof {}".format(model_A_name), color=model_A_color)
    # ax.set_title('sync = {}'.format(sync))

    fig.set_tight_layout(True)
    fig.savefig(os.path.join(
        args.output_dir, 'job_arrival_and_queued_kernels.jpg'),
        bbox_inches='tight')

    # CDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Output CDF
    mask = df['PID'] == model_A_pid
    x, y = cdf(df[mask]['Kernel Dur (ns)'])
    ax = axes[0]
    ax.set_title(f"{model_B_name} Kernel Execution Duration")
    ax.set_xlabel('Kernel Exection Time (ms)')
    ax.set_ylabel('CDF')
    ax.plot(x / SEC_IN_NS * SEC_IN_MS, y, marker='.')
    ax.set_ylim(0, 1.1)


    mask = df['PID'] == model_B_pid
    x, y = cdf(df[mask]['Kernel Dur (ns)'])
    ax = axes[1]
    ax.set_title(f"{model_A_name} Kernel Execution Duration")
    ax.set_xlabel('Kernel Exection Time (ms)')
    ax.set_ylabel('CDF')
    ax.plot(x / SEC_IN_NS * SEC_IN_MS, y, marker='.')
    ax.set_ylim(0, 1.1)

    fig.set_tight_layout(True)
    fig.savefig(os.path.join(args.output_dir, 'cdf.jpg'), bbox_inches='tight')

if __name__ == '__main__':
    main()
