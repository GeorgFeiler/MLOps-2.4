from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "gera",
    "start_date": dt.datetime(2024, 8, 4),
    "retries": 5,
    "retry_delay": dt.timedelta(seconds=10),
    "depends_on_past": False
}

with DAG(
    dag_id='skill_factory_cars_pipeline',
    default_args=args,
    schedule_interval='@daily',
    tags=['skill factory', 'MLOps-2.4', 'cars'],
) as dag:
    
    get_data = BashOperator(
        task_id='get_data',
        bash_command="python3 /opt/clearml/projects/MLOps-2.4/scripts/get_data.py",
        dag=dag
    )
    
    get_features = BashOperator(
        task_id='get_features',
        bash_command="python3 /opt/clearml/projects/MLOps-2.4/scripts/get_features.py /opt/clearml/projects/MLOps-2.4/datasets/raw/cars.csv /opt/clearml/projects/MLOps-2.4/datasets/features/cars.csv",
        dag=dag
    )
    
    change_text_to_numeric = BashOperator(
        task_id='change_text_to_numeric',
        bash_command="python3 /opt/clearml/projects/MLOps-2.4/scripts/change_text_to_numeric.py /opt/clearml/projects/MLOps-2.4/datasets/features/cars.csv /opt/clearml/projects/MLOps-2.4/datasets/prepared/cars.csv",
        dag=dag
    )
    
    train_test_split = BashOperator(
        task_id='train_test_split',
        bash_command="python3 /opt/clearml/projects/MLOps-2.4/scripts/train_test_split.py /opt/clearml/projects/MLOps-2.4/datasets/prepared/cars.csv",
        dag=dag
    )
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command="python3 /opt/clearml/projects/MLOps-2.4/scripts/dt.py /opt/clearml/projects/MLOps-2.4/datasets/train_test_split/cars_train.csv",
        dag=dag
    )
    
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command="python3 /opt/clearml/projects/MLOps-2.4/scripts/evaluate.py /opt/clearml/projects/MLOps-2.4/datasets/train_test_split/cars_test.csv /opt/clearml/projects/MLOps-2.4/models/model.pkl",
        dag=dag
    )
    
    get_data >> get_features >> change_text_to_numeric >> train_test_split >> train_model >> evaluate_model
