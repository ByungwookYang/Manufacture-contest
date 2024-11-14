import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlflow.tracking import MlflowClient

# MLflow Tracking Server URI 설정


def evaluate_and_save_model(model, metrics, threshold=0.8):
    f1 = list(metrics.values())[0]
    accuracy = list(metrics.values())[1]
    precision = list(metrics.values())[2]
    recall = list(metrics.values())[3]

    # F1 스코어가 기준 이상인 경우에만 모델 저장
    if f1 >= threshold:
        print("F1 스코어 기준을 충족하여 모델을 저장합니다.\n")
        print("잠시만 기다려주세요...")

        with mlflow.start_run():
            # 평가 메트릭 로깅
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(model, "model")

        print("모델 저장이 완료되었습니다.")
    else:
          print("F1 스코어 기준을 충족하지 못하여 모델을 저장하지 않습니다.")   
    # 평가 메트릭 반환
    return {
        "f1_score": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

import mlflow
from mlflow.tracking import MlflowClient

# MLflow Client 생성
client = MlflowClient()

# 가장 최근의 run_id를 가져오는 함수
def get_latest_run_id(experiment_id="0"):
    # 가장 최근에 완료된 run 가져오기
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
    if runs:
        return runs[0].info.run_id
    else:
        return None

# 모델 로딩 함수
def load_latest_model():
    latest_run_id = get_latest_run_id()
    if latest_run_id:
        # 가장 최근의 모델 불러오기
        model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
        print(f"모델 {latest_run_id} 로드 완료!")
        return model
    else:
        print("최근에 저장된 모델이 없습니다.")
        return None