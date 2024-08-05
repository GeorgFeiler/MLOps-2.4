# MLOps-2 - Практическое задание №4

## Постановка задачи
* Вам необходимо выбрать один из пройденных инструментов автоматизации процесса машинного обучения (DVC/ClearML / MLFlow / Airflow), реализовать с его помощью процесс обучения любой выбранной вами (или созданной самостоятельно) модели. В зависимости от используемого инструмента определить решаемые экспериментом задачи - работа с различными датасетами, либо подбор оптимальных гиперпараметров, либо сравнение результатов обработки данных различными алгоритмами, либо др. Пояснить цель и ход своих экспериментов.

## Решение задачи
Для выполнения данной задачи был взят датасет ["Cars for sale in Moldova"](https://www.kaggle.com/datasets/alexandrududa/cars-moldova?select=cars.csv). В качестве инструментов были выбраны MLFlow и Airflow. Загружен датасет был путём применения [скрипта](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/scripts/get_data.py). Путём [преобразований](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/scripts/get_features.py) набор данных был урезан с 9 до 6 столбцов, все значения признаков были [приведены к числовому формату](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/scripts/change_text_to_numeric.py) и результат разделён на [тренировочную и тестовую выборки](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/scripts/train_test_split.py). На тренировочном наборе была [обучена модель](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/scripts/dt.py) и [проверена](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/scripts/evaluate.py) на тестовой выборке. Изменение параметров производилось в файле [params.yaml](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/scripts/params.yaml). Конвейер приводился в действие DAG-файлом [cars_prediction_score.py](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/airflow/dags/cars_prediction_score.py). Оценка сохраняемой предсказательной точности сохранялась в файле [evaluate.json](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/evaluate/evaluate.json). Последний оптимистичный результат составлил 0.9747967479674797.

## Технологии

* Oracle VirtualBox 7.0.18
* Ubuntu 24.04 LTS
* Python 3.9.19
* AirFlow 2.9.3
* MLflow 2.14.2 (включая MLflow Tracking Server)
* Visual Studio Code 1.90.2

## Библиотеки

[requirements.txt](https://github.com/GeorgFeiler/MLOps-2.4/blob/main/requirements.txt)

* aiohttp==3.9.5
* aiohttp-retry==2.8.3
* aiosignal==1.3.1
* apache-airflow==2.9.3
* attrs==23.2.0
* diskcache==5.6.3
* distro==1.9.0
* dulwich==0.22.1
* fsspec==2024.6.1
* funcy==2.0
* joblib==1.4.2
* mlflow==2.14.2
* numpy==1.21.6
* pandas==1.5.0
* pathspec==0.12.1
* pygit2==1.15.0
* python-dateutil==2.9.0.post0
* pytz==2024.1
* PyYAML==6.0
* pyyoutube==0.8.1 
* requests==2.32.3
* scikit-learn==1.1.3
* scipy==1.9.3
* tqdm==4.66.4
