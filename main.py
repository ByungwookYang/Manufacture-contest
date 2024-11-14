from b_analysis.database import csv2db, making_dataframe_main_db, create_new_table
from b_analysis.feature_engineering import preprocessing_pipeline_train, preprocessing_pipeline_test, load_columns_and_filter_test_data, save_columns_to_feature_store
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sqlite3
from c_modeling.model_training import find_best_params, train_ensemble_model
from c_modeling.model_validation import validate_model
from c_modeling.model_evaluation import evaluate_and_save_model
from mlflow.tracking import MlflowClient
import mlflow
from c_modeling.model_pipeline import model_training_pipeline

def main():
    # Step 1: CSV 파일을 데이터베이스에 로드 (처음 실행 시 한 번만)
    print("1단계: CSV 파일을 데이터베이스에 로드 중...")
    csv2db()
    print("")

    # Step 2: 데이터베이스에서 데이터 불러오기
    print("2단계: train 데이터 불러오기...")
    train_data = making_dataframe_main_db("train_table")
    print("")

    # Step 3: 데이터 전처리 (NULL 제거, 클러스터링 기반 대체, 특정 열 제거 등)
    print("3단계: 데이터 전처리 중...")
    processed_train = preprocessing_pipeline_train.fit_transform(train_data)
    
    # 전처리 완료된 train 컬럼정보 DB에 저장
    save_columns_to_feature_store(processed_train,"cols")
    print("")

    # Step 6: 전처리 완료 데이터를 DB에 저장
    print("6단계: 전처리완료. 데이터 DB에 저장 중...")
    create_new_table(processed_train,"processed_train")
    print("")

    # Step 7: 전처리 완료된 훈련데이터 불러오기 (모델 훈련/검증용)
    print("7단계: 훈련데이터 불러오기...")
    processed_train = making_dataframe_main_db("processed_train")
    print("")

    # Step 8: 모델 학습 파이프라인 실행
    print("8단계 : 모델 학습 파이프라인 실행 중...")
    metrics_result = model_training_pipeline(processed_train)
    print("최종 메트릭 결과:", metrics_result)

if __name__ == "__main__":
    main()
