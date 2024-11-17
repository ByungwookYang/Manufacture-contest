import os
from b_analysis.database import making_dataframe_main_db
import os
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 데이터프레임 로드
df = making_dataframe_main_db('processed_train')

# 한글 열 이름을 영어로 변경
df['working'] = df['working'].replace({'가동': 'running', '정지': 'stopped'})
# ChatOpenAI LLM 인스턴스 생성
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', api_key=os.environ['OPENAI_API_KEY'])

# LangChain Agent 생성
agent = create_pandas_dataframe_agent(
    llm=llm,
    df = df,
    verbose = True,
    agent_type = AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code = True
)

agent.run('데이터의 행과 열의 갯수는 어떻게 돼?')