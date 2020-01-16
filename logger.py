import os
import json
import csv
import tabulate


def save_json(d, filename):
    """ write dict of data to json file """
    with open(filename, 'w') as f:
        json.dump(d, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_csv(d, filename):
    """ write dict to csv output """
    assert isinstance(d, dict), 'Must pass dict to csv writer'
    exists = os.path.exists(filename)
    with open(filename, 'a') as f:
        w = csv.DictWriter(f, d.keys())
        if not exists: w.writeheader()
        w.writerow(d)
