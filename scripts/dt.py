import sys
import os
import yaml
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model_ml")

params = yaml.safe_load(open("/opt/clearml/projects/MLOps-2.4/scripts/params.yaml"))["cars"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython dt.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = '/opt/clearml/projects/MLOps-2.4/models/model.pkl'
os.makedirs(os.path.dirname(f_output), exist_ok=True)

p_seed = params["seed"]
p_max_depth = params["max_depth"]

df = pd.read_csv(f_input)
X = df.iloc[:, 1:]  # Используем все колонки, кроме первой, как признаки
y = df.iloc[:, 0]   # Используем первую колонку как метки

clf = DecisionTreeClassifier(max_depth=p_max_depth, random_state=p_seed)

with mlflow.start_run():
    clf.fit(X, y)
    mlflow.sklearn.log_model(clf, "model")
    with open(f_output, "wb") as fd:
        pickle.dump(clf, fd)
    mlflow.log_artifact(f_output)
    mlflow.log_artifact('/opt/clearml/projects/MLOps-2.4/scripts/dt.py')
    mlflow.end_run()
