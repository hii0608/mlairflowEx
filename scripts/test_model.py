import mlflow
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


PROJECT_PATH = "/home/dblab/haeun/mlops/ml_airflow/ex1"
df = pd.read_csv(f'{PROJECT_PATH}/datasets/data_test.csv', header=None)
df.columns = ['id', 'counts']
 
model = LinearRegression()
with open(f'{PROJECT_PATH}/models/data.pickle', 'rb') as f:
    model = pickle.load(f)

# MLflow 사용 시작
mlflow.set_tracking_uri("http://localhost:8088")  # 적절한 MLflow 서버 URI 설정
mlflow.set_experiment("test_model")

with mlflow.start_run():
    score = model.score(df['id'].values.reshape(-1, 1), df['counts'])
    print("score=", score)
    
    # MLflow에 테스트 결과 기록
    mlflow.log_metric("test_score", score)
    
    mlflow.log_artifact(f'{PROJECT_PATH}/scripts/test_model.py', artifact_path="test_model code")


# from sklearn.linear_model import LinearRegression
# import pickle
# import pandas as pd



# PROJECT_PATH = "/home/dblab/haeun/mlops/ml_airflow/ex1"
# df = pd.read_csv('{PROJECT_PATH}/datasets/data_test.csv', header=None)
# df.columns = ['id', 'counts']
 
# model = LinearRegression()
# with open('{PROJECT_PATH}/models/data.pickle', 'rb') as f:
#     model = pickle.load(f)
 
# score = model.score(df['id'].values.reshape(-1,1), df['counts'])
# print("score=", score)