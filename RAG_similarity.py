import pandas as pd
from langchain_chroma import Chroma  # 벡터 스토어 가져오기
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # 임베딩 모델
from b_analysis.database import making_dataframe_main_db
from tqdm import tqdm  # 진행 상황 표시
import json
from scipy.spatial.distance import cosine

data_test = making_dataframe_main_db('processed_train').iloc[1550:1600, :]  # 일부 데이터만 로드
df = data_test
json_data = df.to_json(orient="records", force_ascii=False)
data = json.loads(json_data)
text_data = [" ".join([f"{key}: {value}" for key, value in record.items()]) for record in data]
model_name = 'jhgan/ko-sbert-nli'
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': "cpu"},
    encode_kwargs={'normalized_embeddings': True})

# 임베딩 모델 설정 및 임베딩 생성
model_name = 'jhgan/ko-sbert-nli'
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': "cpu"},
    encode_kwargs={'normalized_embeddings': True}
)

# 미리 생성된 임베딩 벡터
print("Creating embeddings...")
embeddings = [hf.embed_query(text) for text in text_data]

# 유사도 계산 함수
def find_similar_texts(query_text, text_data, embeddings, top_k=10):
    query_embedding = hf.embed_query(query_text)
    
    # 각 텍스트와의 유사도 계산
    similarities = []
    for idx, embedding in enumerate(embeddings):
        similarity = 1 - cosine(query_embedding, embedding)  # 코사인 유사도 계산
        similarities.append((text_data[idx], similarity))
    
    # 유사도 순으로 정렬하고 상위 k개 반환
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# 쿼리 텍스트와 유사한 항목 검색
query_text = "passorfail: 1.0 and 불량의 주요 원인"

# 유사도 검색 수행
similar_texts = find_similar_texts(query_text, text_data, embeddings)

# 결과 출력
print("Top similar texts:")
for text, similarity in similar_texts:
    print(f"Text: {text}\nSimilarity: {similarity}\n")


