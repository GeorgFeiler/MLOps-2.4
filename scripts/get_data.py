import os
import kaggle
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/opt/clearml/projects/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data_ml")

def download_kaggle_dataset():
    # Устанавливаем рабочую директорию для скачивания
    download_dir = '/opt/clearml/projects/MLOps-2.4/datasets/raw'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Указываем идентификатор датасета
    dataset = 'alexandrududa/cars-moldova'

    # Скачиваем файл cars.csv
    kaggle.api.dataset_download_file(dataset, 'cars.csv', path=download_dir)

    # Разархивируем скачанные файлы
    os.system(f'unzip -o {download_dir}/cars.csv.zip -d {download_dir}')

    # Удалим zip-файлы после разархивирования
    os.remove(f'{download_dir}/cars.csv.zip')

if __name__ == '__main__':
    with mlflow.start_run():
        download_kaggle_dataset()
        mlflow.log_artifact('/opt/clearml/projects/MLOps-2.4/scripts/get_data.py')
        mlflow.end_run()
