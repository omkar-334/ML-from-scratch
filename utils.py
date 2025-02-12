import os

import kaggle
from sklearn.metrics import accuracy_score, classification_report


def metrics(y_pred, y_true, target_names=None):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))


def download_kaggle_dataset(dataset_name):
    download_path = os.path.join("data", dataset_name.split("/")[-1])
    os.makedirs(download_path, exist_ok=True)

    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    return [os.path.join(download_path, file) for file in os.listdir(download_path)]
