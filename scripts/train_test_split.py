import yaml
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_test_split_ml")

params = yaml.safe_load(open("/opt/clearml/projects/MLOps-2.4/scripts/params.yaml"))["split"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 train_test_split.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output_train = '/opt/clearml/projects/MLOps-2.4/datasets/train_test_split/cars_train.csv'
f_output_test = '/opt/clearml/projects/MLOps-2.4/datasets/train_test_split/cars_test.csv'
os.makedirs(os.path.dirname(f_output_train), exist_ok=True)
os.makedirs(os.path.dirname(f_output_test), exist_ok=True)

p_split_ratio = params["split_ratio"]

df = pd.read_csv(f_input)

# Разделение данных на признаки и метки
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Проверка распределения классов
class_counts = y.value_counts()
classes_to_remove = class_counts[class_counts < 2].index

# Удаление классов с менее чем двумя элементами
for cls in classes_to_remove:
    mask = y != cls
    X = X[mask]
    y = y[mask]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_split_ratio, stratify=y)

# Объединение меток и признаков в один DataFrame для train и test наборов
train_data = pd.concat([y_train, X_train], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

# Сохранение данных с заголовками
train_data.to_csv(f_output_train, header=True, index=False)
test_data.to_csv(f_output_test, header=True, index=False)

if __name__ == '__main__':
    with mlflow.start_run():
        mlflow.log_artifact(f_output_train)
        mlflow.log_artifact(f_output_test)
        mlflow.log_artifact('/opt/clearml/projects/MLOps-2.4/scripts/train_test_split.py')
        mlflow.end_run()
