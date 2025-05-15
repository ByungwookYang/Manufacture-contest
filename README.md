# MLOps와 LLM 기반 실시간 제조 공정 모니터링 및 불량 관리 플랫폼
[대회 참고](https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=16)

## 📌 **공모전 개요**

인공지능 중소벤처 제조플랫폼(KAMP) 내 등재된 제조AI데이터셋을 활용하여 중소 제조기업이 직면할 수 있는 공통문제를 해결 또는 개선할 수 있는 우수한 인공지능 분석 모델 개발 

**해결 과제 :** 뿌리업종(주조) 기업의 생산성 향상 및 작업환경 개선을 위한 아이디어를 제시하고 인공지능 알고리즘으로 구현

 

---
## 📣 **공모전 정보**

- **대회명 :** 2024년 제4회 K-인공지능(AI) 제조데이터 분석 경진대회
- **주최기관 :** 중소벤처기업부, 스마트제조혁신추진단, KAIST
- **진행기간 :** 20241002 - 20241127
- **수상 :** 장려상 (6th / 189teams)

---

## **❗ 문제 상황**
<img width="548" alt="image" src="https://github.com/user-attachments/assets/dd428f55-e661-41b8-bce3-1c6274d57c75" />


---

## 🧠 **구현 아디이어**
- 생산성 향상 및 작업환경 개선을 위해 LLM 시스템 도입
- LLM 챗봇 시스템을 이용해 불량으로 예측된 데이터에 대한 설명 요약
- MLOPs를 이용한 유지보수 및 자동화

---

## ⚙️ 구현 과정

### 1) 데이터를 train/test 7:3 split

<img width="602" alt="image" src="https://github.com/user-attachments/assets/93193fe0-0a18-4368-9e04-c0e3b6a13f18" />

### 2) CatBoost Modeling

<img width="657" alt="image" src="https://github.com/user-attachments/assets/531c53ac-fc3b-4119-a018-b8d79c1fbd39" />

### 3) Level1 MLOPs 구현

<img width="449" alt="image" src="https://github.com/user-attachments/assets/03b5e5bd-a415-4d4d-8e40-721bf2f0e212" />

### 4) 예측된 불량 원인 분석 LLM

<img width="676" alt="image" src="https://github.com/user-attachments/assets/c323c192-1567-4221-bf5b-4e0d326d75bf" />

### 자세한 과정은 발표자료에 정리되어 있습니다.
[경진대회_본선발표자료_fianl.pdf](https://github.com/user-attachments/files/17793440/_._fianl.pdf)

---
## 💻 **구현**
<img width="1031" alt="image" src="https://github.com/user-attachments/assets/df22cfc4-c50b-4e7b-9c0c-10b63de322fd" />

<img width="1028" alt="image" src="https://github.com/user-attachments/assets/87baf6e0-2d29-4f92-9218-c8af5cf75643" />

<img width="1027" alt="image" src="https://github.com/user-attachments/assets/646f7d55-9b35-49ed-8a3c-7835988240c0" />

<img width="1019" alt="image" src="https://github.com/user-attachments/assets/fdaa82e7-30cc-4215-b1b5-45bad4874e25" />

<img width="1020" alt="image" src="https://github.com/user-attachments/assets/9c5f4cdb-8ac1-4b3d-8d95-3adb890daa3f" />

<img width="1021" alt="image" src="https://github.com/user-attachments/assets/74829c66-f741-4545-a535-6f91f5441ab5" />

<img width="1031" alt="image" src="https://github.com/user-attachments/assets/b1ba8e9f-1a02-4215-b02f-777fed7ce5f1" />


---

## 💡 **분석 의의**
- AI와 MLOps를 활용해 불량을 실시간으로 감지하고, 주요 요인을 빠르게 파악가능

- 다양한 제조 업종으로 확장 가능하며, 자동화된 품질 관리 환경 구축이 가능

- MLOps 고도화와 LLM 기반 프롬프트 엔지니어링을 통해 더 정밀한 원인 분석과 대응이 가능




--- 
## 직접 구현 방법
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





