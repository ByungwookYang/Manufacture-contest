import sys
import os

# 'project' 폴더의 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from b_analysis.database import making_dataframe_main_db, create_new_table
from b_analysis.feature_engineering import preprocessing_pipeline_test, load_columns_and_filter_test_data
import streamlit as st
import pandas as pd
import time
import json
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from c_modeling.model_evaluation import get_latest_run_id, load_latest_model
from b_analysis.feature_engineering import preprocess_test

# 성능 평가 지표 초기화
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'actuals' not in st.session_state:
    st.session_state.actuals = []

# 성능 지표 시각화를 위한 placeholder 설정
metrics_placeholder = st.empty()


# 모델 배포
model = load_latest_model()

# 실시간 스트리밍처럼 작동하기
def simulate_real_time_data_streaming(table_name):
    # DB에서 test데이터 불러오기
    test_data = making_dataframe_main_db(table_name)
    processed_test = preprocessing_pipeline_test.fit_transform(test_data)
    filtered_test = load_columns_and_filter_test_data("cols", processed_test)
    X, categorical_cols  = preprocess_test(filtered_test)
    
    positives_placeholder = st.empty()
    negatives_placeholder = st.empty()
    chart_placeholder = st.empty()

    if 'combined_predictions' not in st.session_state:
        st.session_state.combined_predictions = pd.DataFrame()
    if 'positives' not in st.session_state:
        st.session_state.positives = pd.DataFrame()
    if 'negatives' not in st.session_state:
        st.session_state.negatives = pd.DataFrame()

    for i in range(st.session_state.get('start_index', 0), len(X)):
        if st.session_state.get('stop_clicked'):
            st.write("예측이 중단되었습니다.")
            break

        # 예측 수행 및 결과 저장
        prediction = model.predict(X.iloc[[i]])
        X.at[X.index[i], "Prediction"] = prediction[0]
        
        # 세션 상태에 예측과 실제값 저장
        st.session_state.predictions.append(prediction[0])
        

        # 예측 결과 시각화
        if prediction[0] == 1:
            st.session_state.positives = pd.concat([st.session_state.positives, X.iloc[[i]]], ignore_index=True)
        else:
            st.session_state.negatives = pd.concat([st.session_state.negatives, X.iloc[[i]]], ignore_index=True)

        st.session_state.combined_predictions = pd.concat([st.session_state.combined_predictions, X.iloc[[i]]], ignore_index=True)

        chart_data = pd.DataFrame({"Prediction": st.session_state.combined_predictions["Prediction"]})
        chart_placeholder.line_chart(chart_data)
        positives_placeholder.write("### Prediction = 1 데이터")
        positives_placeholder.dataframe(st.session_state.positives)
        negatives_placeholder.write("### Prediction = 0 데이터")
        negatives_placeholder.dataframe(st.session_state.negatives)

        st.session_state.start_index = i + 1
        time.sleep(3)
    
    if st.session_state.start_index >= len(X):
        st.session_state.simulation_running = False
        st.session_state.stop_clicked = False
        st.write("All data predicted.")



