import pandas as pd


PROJECT_PATH = "/home/dblab/haeun/mlops/ml_airflow/ex1"
df = pd.read_csv(f'{PROJECT_PATH}/datasets/data.csv', header=None)
 
df[0] = (df[0]-df[0].min())/(df[0].max()-df[0].min())
 
with open(f'{PROJECT_PATH}/datasets/data_processed.csv', 'w') as f:
    for i, item in enumerate(df[0].values):
        f.write("{},{}\n".format(i, item))