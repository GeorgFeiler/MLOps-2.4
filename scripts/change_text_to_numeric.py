import sys
import os
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_features_ml")

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 get_features.py input-file output-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = sys.argv[2]
os.makedirs(os.path.dirname(f_output), exist_ok=True)

def process_data(input_path, output_path):
    # Чтение данных
    cars = pd.read_csv(input_path)
    
    # Пример обработки данных: выбираем только некоторые столбцы и сохраняем их
    selected_columns = ['Make', 'Model', 'Year', 'Distance', 'Fuel_type', 'Price(euro)']
    cars = cars[selected_columns]
    
    # Сохранение данных в новый файл
    cars.to_csv(output_path, index=False)

# Выполнение обработки данных
if __name__ == '__main__':
    with mlflow.start_run():
        process_data(f_input, f_output)
        mlflow.log_artifact(f_output)
        mlflow.log_artifact('/opt/clearml/projects/MLOps-2.4/scripts/get_features.py')
        mlflow.end_run()
