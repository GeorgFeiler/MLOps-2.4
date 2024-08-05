import os
import sys
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("evaluate_model_ml")

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py data-file model\n")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
X = df.iloc[:, 1:]  # Используем все колонки, кроме первой, как признаки
y = df.iloc[:, 0]   # Используем первую колонку как метки

with open(sys.argv[2], "rb") as fd:
    clf = pickle.load(fd)

# Предсказание и оценка
y_pred = clf.predict(X)
score = accuracy_score(y, y_pred)
print("accuracy =", score)

prc_file = '/opt/clearml/projects/MLOps-2.4/evaluate/evaluate.json'
os.makedirs(os.path.dirname(prc_file), exist_ok=True)

with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)

if __name__ == '__main__':
    with mlflow.start_run():
        mlflow.log_metric("accuracy", score)
        mlflow.log_artifact(prc_file)
        mlflow.log_artifact('/opt/clearml/projects/MLOps-2.4/scripts/evaluate.py')
        mlflow.end_run()
