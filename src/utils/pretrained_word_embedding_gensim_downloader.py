import os

from tqdm import tqdm
import gensim.downloader as api

from consts import GENSIM_MODEL_NAME_TO_PATH


def download_model(model_name, model_dir):
    print(f'\nDownloading model: {model_name}')
    model = api.load(model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, model_name + ".kv"))


def download_all_models():
    for model_name, model_dir in tqdm(GENSIM_MODEL_NAME_TO_PATH.items(), desc="Model downloading loop"):
        download_model(model_name, model_dir)


if __name__ == '__main__':
    download_all_models()