from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from c_modeling.model_training import find_best_params, train_ensemble_model
from c_modeling.model_validation import validate_model
from c_modeling.model_evaluation import evaluate_and_save_model
from b_analysis.feature_engineering import preprocess_data

# 모델 학습 파이프라인 구성
def model_training_pipeline(processed_train):
    # Step 8: 하이퍼파라미터 탐색 및 최적 파라미터 찾기
    print("8단계: 하이퍼파라미터 탐색 중...")
    best_params = find_best_params(processed_train)  # find_best_params의 결과를 best_params에 저장

    # Step 9: 최적 파라미터로 모델 학습
    print("9단계: 모델 학습 중...")
    voting_model = train_ensemble_model(processed_train, best_params=best_params)  # best_params를 전달하여 모델 학습

    # Step 10: 모델 검증 및 조건부 저장
    print("10단계: 모델 검증 및 저장 중...")
    metrics = validate_model(voting_model, processed_train)
    metrics_result = evaluate_and_save_model(voting_model, metrics, threshold=0.8)

    print("최종 평가 메트릭:", metrics_result)
    return metrics_result