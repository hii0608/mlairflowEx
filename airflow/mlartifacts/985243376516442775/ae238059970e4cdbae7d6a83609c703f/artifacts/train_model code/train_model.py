from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import os
 
import mlflow
from mlflow.tracking import MlflowClient


PROJECT_PATH = "/home/dblab/haeun/mlops/ml_airflow/ex1"

mlflow.set_tracking_uri("http://localhost:8088") 
mlflow.set_experiment("train_model")
 
df = pd.read_csv(f'{PROJECT_PATH}/datasets/data_train.csv', header=None)
df.columns = ['id', 'counts']
model = LinearRegression()
 
with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path=f"{PROJECT_PATH}/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()
 
model.fit(df['id'].values.reshape(-1,1), df['counts'])
 
with open(f'{PROJECT_PATH}/models/data.pickle', 'wb') as f:
    pickle.dump(model, f)