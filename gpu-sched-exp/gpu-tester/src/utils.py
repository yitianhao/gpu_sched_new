import collections
import json
import time
import numpy as np
import yaml
import pandas as pd

def read_json_file(filename):
    with open(filename, 'r') as f:
        content = json.load(f, object_pairs_hook=collections.OrderedDict)
    return content

def write_json_file(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)
        f.flush()

def read_yaml_file(filename):
    with open(filename, 'r') as f:
        content = yaml.safe_load(f)
    return content

def sem(data):
    return np.std(data) / np.sqrt(len(data))

def parse_log(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        return df[1:]
    raise ValueError("Unrecoginized file extension.")
    # drop the first jct which is high due to cold start


    # elif filename.endswith(".log"):
    #     jcts = []
    #     with open(filename, 'r') as f:
    #         for line in f:
    #             # Strips the newline character
    #             cols = line.rstrip('\n').split(" ")
    #             if len(cols) == 4 and cols[0] == "JCT":
    #                 jcts.append(float(cols[3]))


def get_jcts_from_profile(profile_filename):
    profile = read_yaml_file(profile_filename)
    jcts = [row['jct'] for row in profile if row['jct'] < 1000]
    return jcts


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')
