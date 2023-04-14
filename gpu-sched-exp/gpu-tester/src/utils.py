import collections
import json
import numpy as np
import yaml

def read_json_file(filename):
    with open(filename, 'r') as f:
        content = json.load(f, object_pairs_hook=collections.OrderedDict)
    return content

def read_yaml_file(filename):
    with open(filename, 'r') as f:
        content = yaml.safe_load(f)
    return content

def sem(data):
    return np.std(data) / np.sqrt(len(data))

def parse_log(filename):
    jcts = []
    with open(filename, 'r') as f:
        for line in f:
            # Strips the newline character
            cols = line.rstrip('\n').split(" ")
            if len(cols) == 4 and cols[0] == "JCT":
                jcts.append(float(cols[3]))
    return jcts[1:]


def get_jcts_from_profile(profile_filename):
    profile = read_yaml_file(profile_filename)
    jcts = [row['jct'] for row in profile if row['jct'] < 1000]
    return jcts
