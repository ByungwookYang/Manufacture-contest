import sys
import os
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from streamlit_chat import message # type: ignore

# 'project' 폴더의 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from b_analysis.database import making_dataframe_main_db, create_new_table, update_main_db
from d_deployment.dash_class import simulate_real_time_data_streaming
from c_modeling.model_evaluation import load_latest_model
from b_analysis.feature_engineering import preprocessing_pipeline_train
from c_modeling.model_pipeline import model_training_pipeline
# 다른 파일에서 agent와 conversation 사용하기
from lang_chain_class import create_agent

# agent와 conversation 객체 생성
agent, conversation = create_agent()

# Page title and styling
st.markdown("""
    <style>
    .title {
        text-align: left;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    <div class="title">Interactive Defective Data Prediction Dashboard</div>
    """, unsafe_allow_html=True)

# Load the model
model = load_latest_model()

# OpenAI API Key 설정
openai_api_key = st.secrets["openai"]["api_key"]

# LangChain 초기화
if "conversation" not in st.session_state:
    memory = ConversationBufferMemory(return_messages=True)  # 대화의 맥락을 저장
    st.session_state["messages"] = []  # 메시지 상태 초기화

    # ChatOpenAI 객체 설정
    chat = ChatOpenAI(
        model_name="gpt-4o", 
        openai_api_key=openai_api_key,
        temperature=0.8,
        streaming=True  # 실시간 스트리밍 활성화
    )

    # 역할을 지정하는 프롬프트 템플릿 생성
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=(
    "You are an expert in manufacturing defect analysis. Follow a step-by-step reasoning approach to identify causes "
    "and suggest improvements. Each step should consider variables like temperature, pressure, and cooling time. "
    "Provide explanations that emphasize technical accuracy and practical insights to enhance manufacturing processes.\n\n"
    "{history}\nUser: {input}\nAssistant:"
),
    )

    # ConversationChain 생성
    st.session_state["conversation"] = ConversationChain(
        llm=chat,
        memory=memory,
        prompt=prompt,
    )

# 필요한 세션 상태 초기화
if 'start_index' not in st.session_state:
    st.session_state.start_index = 0
if 'combined_predictions' not in st.session_state:
    st.session_state.combined_predictions = pd.DataFrame()
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'stop_clicked' not in st.session_state:
    st.session_state.stop_clicked = False
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Chatbot 인터페이스
st.sidebar.title("Chat with GPT-4o")

# 사용자 입력을 받으면 모델 응답을 즉시 처리
user_input = st.sidebar.text_input("You:", "", key="user_input")

if user_input:
    # 사용자 메시지 저장
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # LangChain의 ConversationChain을 사용하여 대화 생성
    # agent를 사용하여 응답을 생성
    response_from_agent = agent.run(user_input)  # agent를 사용하여 응답 받기
    
    # 응답 메시지 저장
    st.session_state["messages"].append({"role": "assistant", "content": response_from_agent})
    st.session_state.generated.append(response_from_agent)

# Display chatbot messages
if st.session_state["messages"]:
    for i in range(len(st.session_state["messages"]) - 1, -1, -1):
        role = st.session_state["messages"][i]["role"]
        content = st.session_state["messages"][i]["content"]
        message(content, is_user=(role == "user"), key=f"{role}_{i}")

# Real-time simulation control buttons
start_button = st.button("Simulate Real-time Predictions")
stop_button = st.button("Stop Predictions")

if start_button and not st.session_state.simulation_running:
    st.session_state.simulation_running = True
    st.session_state.stop_clicked = False
    st.write("Real-time Prediction Simulation Started!")

if stop_button and st.session_state.simulation_running:
    st.session_state.stop_clicked = True
    st.session_state.simulation_running = False

# Display warning if simulation is running
if st.session_state.simulation_running:
    st.warning("Real-time predictions in progress.")
else:
    st.empty()

# Trigger real-time simulation
if st.session_state.simulation_running:
    simulate_real_time_data_streaming('test_table')

# Post-simulation options
if not st.session_state.simulation_running and st.session_state.stop_clicked:
    st.write("Choose an action:")
    col1, col2, col3 = st.columns(3)
    with col1:
        retrain_option = st.button("Retrain Model")
    with col2:
        view_data_option = st.button("View Prediction Data")
    with col3:
        resume_option = st.button("Resume Predictions")

    if retrain_option:
        st.write("Retraining Model...")
        
        # 기존 학습 데이터 로드
        train_data = making_dataframe_main_db("train_table")

        # 누적된 예측 데이터와 결합
        new_train_data = pd.concat([train_data, st.session_state.combined_predictions], ignore_index=True)

        # 데이터베이스 업데이트
        update_main_db('train_table', new_train_data)
        
        train = making_dataframe_main_db('train_table')

        # 트레인 데이터 전처리
        processed_train = preprocessing_pipeline_train.fit_transform(train)
        
        # 모델 학습 및 Optuna로 하이퍼파라미터 최적화
        metrics_result = model_training_pipeline(processed_train)
    
        # 세션 상태 초기화
        st.session_state.combined_predictions = pd.DataFrame()
        st.session_state.start_index = 0
        st.session_state.stop_clicked = False

        st.write("재학습을 완료하였습니다.")

    elif view_data_option:
        st.write("Prediction Results:")
        st.dataframe(st.session_state.combined_predictions)

    elif resume_option:
        st.write("Resuming predictions.")
        st.session_state.simulation_running = True
        st.session_state.stop_clicked = False
        st.warning("Real-time predictions in progress.")

if st.session_state.simulation_running:
    simulate_real_time_data_streaming('test_table')
