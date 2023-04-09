import argparse
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', '--file', metavar="FILEPATH", type=str,
                        help="Path to the csv file of a model parsed from a"
                             "nsys csv. Filename format: "
                             "[MODEL_NAME]_[PID].csv", required=True)
    parser.add_argument('-o', '--output-dir', metavar="OUTPUT_DIR", type=str,
                        help="Path to save the output figures.", default='.')
    args = parser.parse_args()

    fname_tokens = os.path.splitext(os.path.basename(args.file))[0].split('_')
    model_name = '_'.join(fname_tokens[:-1])
    pid = fname_tokens[-1]

    os.makedirs(args.output_dir, exist_ok=True)

    # csv header definition:
    # /opt/nvidia/nsight-systems/2022.4.2/host-linux-x64/reports/kernelexectrace.py: Line 22
    df = pd.read_csv(args.file)
    durs = df['KernelDur']

    # Output CDF
    fig, ax = plt.subplots(figsize=(12, 8))
    length = len(durs)
    x = np.sort(durs)
    # Get the CDF values of y
    y = np.arange(length) / float(length)

    ax.set_title(f"Model {model_name} Kernel Execution Duration CDF")
    ax.set_xlabel('Kernel Exection Time (ns)')
    ax.set_ylabel('CDF')
    ax.plot(x, y, marker='.')

    cdf_path = os.path.join(
        args.output_dir,
        f"{model_name}_{pid}_KernelExecDuration_CDF.jpg")
    fig.savefig(cdf_path)

if __name__ == '__main__':
    main()
