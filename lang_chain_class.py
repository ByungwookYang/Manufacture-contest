import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from b_analysis.database import making_dataframe_main_db

# .env 파일에서 환경 변수 로드
load_dotenv()

def create_agent():
    # 데이터프레임 로드
    df = making_dataframe_main_db('processed_train')

    # 한글 열 이름을 영어로 변경
    df['working'] = df['working'].replace({'가동': 'running', '정지': 'stopped'})

    # ChatOpenAI LLM 인스턴스 생성
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', api_key=os.environ['OPENAI_API_KEY'])

    # LangChain Agent 생성
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    # ConversationBufferMemory 대화의 맥락을 저장하는 메모리 객체 생성
    memory = ConversationBufferMemory(return_messages=True)
    
    # ChatOpenAI 객체 생성 (실시간 스트리밍 설정)
    chat = ChatOpenAI(
        model_name="gpt-4o",  # 예시로 GPT-4로 설정
        openai_api_key=os.environ['OPENAI_API_KEY'],
        temperature=0.8,
        streaming=True
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
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        prompt=prompt,
    )
    
    # 반환하는 객체에 agent와 conversation 포함
    return agent, conversation


# `create_agent`로 agent와 conversation 객체 생성
agent, conversation = create_agent()

# 실제로 agent를 사용하여 데이터프레임을 쿼리하거나 대화할 수 있음
user_input = "What is the current status of the working process?"

# Agent를 사용하여 데이터프레임에서 응답을 얻기
response_from_agent = agent.run(user_input)
print("Agent's response to user input:", response_from_agent)

# Conversation을 사용하여 대화 흐름 관리
response_from_conversation = conversation.predict(input=user_input)
print("Conversation's response:", response_from_conversation)