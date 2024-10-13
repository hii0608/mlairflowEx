import requests
import json
from pyyoutube import Api
import os
 
import mlflow
from mlflow.tracking import MlflowClient
 
 
PROJECT_PATH = "/home/dblab/haeun/mlops/ml_airflow/ex1"
os.environ["MLFLOW_REGISTRY_URI"] = "/home/dblab/haeun/mlops/ml_airflow/ex1/mlruns/" # 원본 repo는 mlflow 인데, 내 환경의 폴더 이름은 mlruns였음
mlflow.set_tracking_uri("http://localhost:8088")
mlflow.set_experiment("get_data") 
 
key = "{YOUR_API_KEY}" #GCP에서 발급받은 youtube dataset 접근 API
api = Api(api_key=key) 
query = "'Mission Impossible'"
video = api.search_by_keywords(q=query, search_type=["video"], count=10, limit=30)
maxResults = 10
nextPageToken = ""
s = 0
 
with mlflow.start_run():
    for i, id_ in enumerate([x.id.videoId for x in video.items]):
        uri = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
              "key={}&textFormat=plainText&" + \
              "part=snippet&" + \
              "videoId={}&" + \
              "maxResults={}&" + \
              "pageToken={}"
        uri = uri.format(key, id_, maxResults, nextPageToken)
        content = requests.get(uri).text
        data = json.loads(content)
        for item in data['items']:
            s += int(item['snippet']['topLevelComment']['snippet']['likeCount'])
    mlflow.log_artifact(local_path= f"{PROJECT_PATH}/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()
 
with open(f"{PROJECT_PATH}/datasets/data.csv", 'a') as f:
    f.write("{}\n".format(s))