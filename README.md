# MLOps와 LLM 기반 실시간 제조 공정 모니터링 및 불량 관리 플랫폼
[대회 참고](https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=16)

### 📌 **공모전 개요**

인공지능 중소벤처 제조플랫폼(KAMP) 내 등재된 제조AI데이터셋을 활용하여 중소 제조기업이 직면할 수 있는 공통문제를 해결 또는 개선할 수 있는 우수한 인공지능 분석 모델 개발 

**해결 과제 :** 뿌리업종(주조) 기업의 생산성 향상 및 작업환경 개선을 위한 아이디어를 제시하고 인공지능 알고리즘으로 구현

 

---
### 📣 **공모전 정보**
- **대회명 :** 2024년 제4회 K-인공지능(AI) 제조데이터 분석 경진대회
- **주최기관 :** 중소벤처기업부, 스마트제조혁신추진단, KAIST
- **진행기간 :** 20241002 - 20241127
- **수상 :** 장려상 (6th / 189teams)

---
### 🧠 **구현 아디이어**



 
# 구현방법
### 0. 환경설정
```{python}
# conda 환경 생성
conda env create -f environment.yml

# conda 환경 활성화
conda activate my_project_env
```
environment.yml 에 사용한 패키지를 모두 저장해두었습니다.
터미널창에 
```{bash}
# conda 환경 생성
conda env create -f environment.yml
```

```{bash}
# conda 환경 활성화
conda activate my_project_env
```

위와 같은 순서로 실행해주세요.
구현한 환경과 동일한 환경셋팅이 완료됩니다.

### 1. openai API 입력하기
.env 파일과 .straemlit 폴더안에 secret.toml 파일에 openai API를 발급받아 입력해주세요.


### 2. 데이터 전처리 및 모델구현, 모델저장
```{bash}
python main.py
```
위와 같이 터미널에서 실행하시면 파이프라인을 통해 연결해둔 전처리부터 모델 저장까지 한번에 되는 것을 확인할 수 있습니다.

자세한 전처리 및 모델링은 b_analysis폴더와 c_modeling 폴더의 .py파일을 확인해주세요.

### 3. 배포 및 대시보드 확인
```{bash}
streamlit run dash.py
```
위와 같이 터미널 창에서 실행하시면, 구현된 대시보드를 확인할 수 있습니다.

자세한 배포과정 및 Lang chain 방법은 d_deployment 파일과 lang_chain_class.py, dash.py에서 확인할 수 있습니다.

# 자세한 과정
[경진대회_본선발표자료_fianl.pdf](https://github.com/user-attachments/files/17793440/_._fianl.pdf)





