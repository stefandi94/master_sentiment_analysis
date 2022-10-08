import json
import os.path
import pickle
from datetime import datetime as dt


def create_dir_path(file_path):
    file_path = os.sep.join(file_path.split(os.sep)[:-1])
    os.makedirs(file_path, exist_ok=True)


def get_time():
    now = dt.now()
    current_time = now.strftime("%d-%m-%y %H-%M-%S")
    return current_time


def save_json(file, path):
    with open(path, "w") as fp:
        json.dump(file, fp)


def load_json(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def save_pickle(data, output_path):
    with open(output_path, "wb") as fp:
        pickle.dump(data, fp)


def load_pickle(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

