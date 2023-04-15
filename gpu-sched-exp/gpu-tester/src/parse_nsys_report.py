import argparse
import collections
import csv
import os
import sys
import numpy as np
import json
from collections import OrderedDict
from datetime import datetime

# Column indexes
COL_API_START = 0
COL_API_DUR = 1
COL_KERNEL_START = 4
COL_KERNEL_DUR = 5
COL_PID = 7


def main():
    # Get input file path
    parser = argparse.ArgumentParser(description="Kernel CSV process script")
    parser.add_argument('-f', '--file', metavar="FILEPATH", help="Specifies the path to the input file", required=True)
    parser.add_argument('-p', '--pids', metavar="FILEPATH", help="Specifies the path to the pid file", required=True)
    args = parser.parse_args()
    filename = args.file
    pids = args.pids

    # Parse Experiment PIDs file
    try:
        with open(pids, 'r') as pids_input:
            experiment_pids = json.load(pids_input, object_pairs_hook=collections.OrderedDict)
    except FileNotFoundError:
        print(f"Input Experiment PIDs file: [{pids}] not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Input Experiment PIDs file: [{pids}] invalid.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    out_dir = os.path.basename(args.file) + f"_{datetime.today().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    out_files = []
    writers = dict()
    num_kernels = OrderedDict()
    kernel_durs = OrderedDict()
    pid_to_modelname = dict()
    try:
        with open(filename, 'r') as file_input:
            reader = csv.reader(file_input)
            next(reader)

            for row in reader:
                # Get PID
                pid = row[COL_PID]
                # Get file for PID
                if pid not in writers:
                    # Find pid's model name
                    for model_name, model_pid in experiment_pids:
                        if int(pid) == model_pid:
                            pid_to_modelname[pid] = model_name
                            break

                    out_file_path = os.path.join(out_dir, f"{pid_to_modelname[pid]}_{pid}.csv")
                    out_file = open(out_file_path, "w")
                    out_files.append(out_file)

                    # Add writer to dict
                    writer = csv.writer(out_file, lineterminator='\n')
                    writers[pid] = writer

                    # Write the header
                    writer.writerow(['IssueStart', 'IssueEnd', 'ExecutionStart', 'ExecutionEnd', 'KernelDur'])

                    # Initialize kernel vars
                    num_kernels[pid] = 0
                    kernel_durs[pid] = []
                else:
                    writer = writers[pid]

                api_start = ensure_int(row[COL_API_START])
                api_dur = ensure_int(row[COL_API_DUR])
                kernel_start = ensure_int(row[COL_KERNEL_START])
                kernel_dur = ensure_int(row[COL_KERNEL_DUR])

                # Write to CSV file
                writer.writerow([
                    api_start,
                    api_start + api_dur,
                    kernel_start,
                    kernel_start + kernel_dur,
                    kernel_dur
                ])

                # Increment num_kernels
                num_kernels[pid] += 1

                # Append kernel duration
                kernel_durs[pid].append(kernel_dur)

        # Output summary log
        summary_path = os.path.join(out_dir, "summary.log")
        with open(summary_path, 'w') as summary_file:
            summary_file.write('NumKernels:\n')
            for pid, num in num_kernels.items():
                summary_file.write(f"{pid_to_modelname[pid]}_{pid} {num}\n")
            summary_file.write('MedianKernelDur:\n')
            for pid, durs in kernel_durs.items():
                summary_file.write(f"{pid_to_modelname[pid]}_{pid} {np.median(durs)}\n")

    except FileNotFoundError:
        # Remove output directory
        os.rmdir(out_dir)
        print(f"Input file: [{filename}] not found.", file=sys.stderr)
        sys.exit(1)
    finally:
        # Close all output files
        for f in out_files:
            f.close()


def ensure_int(value):
    try:
        return int(value)
    except ValueError:
        return 0


if __name__ == '__main__':
    main()
