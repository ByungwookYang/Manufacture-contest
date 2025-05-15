# 2024년 제4회 K-인공지능(AI) 제조데이터 분석 경진대회
[대회 참고](https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=16)

📌 **공모전 개요**

- 주제 : 인공지능 중소벤처 제조플랫폼(KAMP) 내 등재된 제조AI데이터셋을 활용하여 중소 제조기업이 직면할 수 있는 공통문제*를 해결 또는 개선할 수 있는 우수한 인공지능 분석 모델 개발 

- 목적 : 제조 공정에서 불량 데이터가 발생하는 원인은 장비 오류 온도 변화 작업자 실수 재료 품질 문제 등 다양합니다 하지만 수동으로 불량을 역추적하는 방식에는 한계가 있어 즉각적인 조치가 어렵고 동일한 문제가 반복될 위험이 있습니다 .

이를 해결하기 위해 실시간 모니터링 시스템을 도입하여 데이터를 실시간으로 수집하고 분석함으로써 불량 발생을 예측하고 신속하게 감지할 수 있는 환경을 구축했습니다 
예측 모델로는 CatBoost 알고리즘에 더해 앙상블 방법을 이용하여 자동 화된 불량 분류 시스템을 설계하며 안정성을 강화하였고 이를 통해 불량 패턴을 학습하고 불량 여부를 자동으로 분류하도록 했습니다.

 또한 MLOps환경을 통해 데이터 구축 전처리 학습 과정을 자동화하고 관리자가
시스템과 실시간으로 상호작용할 수 있도록 지원했습니다 

더불어 모델인 LLM GPT-3.5 Turbo API 를 로 호출하여 대시보드에 탑재함으로써 불량 데이터의 특징을 이해하고 필요한 조치를 즉각적으로 취할 수 있는 인터페이스를 제공했습니다.

📣 **공모전 정보**
- 대회명 : 2024년 제4회 K-인공지능(AI) 제조데이터 분석 경진대회



 
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





