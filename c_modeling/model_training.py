import mlflow
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import json
from mlflow.tracking import MlflowClient
from b_analysis.feature_engineering import preprocess_data

# optuna 파라미터 범위조정 
def objective(trial, X_train, X_val, y_train, y_val, categorical_cols):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 10),
        'class_weights': {0: trial.suggest_float('class_weights_0', 1.0, 10.0),
                          1: trial.suggest_float('class_weights_1', 1.0, 10.0)},
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'iterations': trial.suggest_int('iterations', 500, 1500)
    }
    model = CatBoostClassifier(**params, random_state=77, verbose=0, cat_features=categorical_cols, early_stopping_rounds=150)
    model.fit(X_train, y_train)
    f1 = f1_score(y_val, model.predict(X_val))
    return f1

# optuna를 이용한 베스트 파라미터 찾기
def find_best_params(processed_train, random_states=[98, 99, 77, 7777], n_trials=1):
    X, y, categorical_cols = preprocess_data(processed_train)
    
    best_params = {}  # 각 random_state에 대해 최적의 하이퍼파라미터 저장

    # 각 random_state마다 Optuna 최적화 수행
    
    for random_state in random_states:
        # 데이터셋을 학습 및 검증용으로 분리
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)

        # Optuna 스터디 생성 및 최적화
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val, categorical_cols), n_trials=n_trials)

        # 최적의 하이퍼파라미터로 모델 학습
        params = {
            'learning_rate': study.best_params['learning_rate'],
            'depth': study.best_params['depth'],
            'class_weights': {
                0: study.best_params['class_weights_0'],
                1: study.best_params['class_weights_1']
            },
            'l2_leaf_reg': study.best_params['l2_leaf_reg'],
            'iterations': study.best_params['iterations']
        }
        
        # 최적의 하이퍼파라미터로 학습된 모델 생성
        model = CatBoostClassifier(**params, random_state=random_state, verbose=0, cat_features=categorical_cols)
        model.fit(X_train, y_train)
        
        # 변수 중요도 추출
        feature_importances = model.get_feature_importance(Pool(X_train, label=y_train, cat_features=categorical_cols))
        feature_names = X_train.columns

        # 변수 중요도를 딕셔너리 형태로 저장
        importance_dict = dict(zip(feature_names, feature_importances))

        # 각 시드에 대한 최적의 하이퍼파라미터와 변수 중요도를 저장
        best_params[random_state] = params

        # MLflow에 하이퍼파라미터와 변수 중요도 기록
        with mlflow.start_run(run_name=f"best_model_random_state_{random_state}"):
            mlflow.log_params(params)  # 하이퍼파라미터 기록
            # 변수 중요도 저장 위치를 mlruns 디렉토리로 지정하여 로그
            mlflow.log_dict(importance_dict, f"feature_importances_{random_state}.json")

    return best_params

# 앙상블모델(soft voting)
def train_ensemble_model(processed_train, best_params):
    X, y, categorical_cols = preprocess_data(processed_train)

    models = []
    for i, (random_state, params) in enumerate(best_params.items()):
        model = CatBoostClassifier(**params, random_state=random_state, cat_features=categorical_cols, verbose=0, early_stopping_rounds=150)
        models.append((f'cat_{i}', model))

    voting_model = VotingClassifier(estimators=models, voting='soft')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1998)
    voting_model.fit(X_train, y_train)
    return voting_model

def load_feature_importances(run_id, random_state):
    """
    주어진 run_id와 random_state에 해당하는 변수 중요도 파일을 MLflow에서 로드
    """
    client = MlflowClient()
    download_path = client.download_artifacts(run_id, f"feature_importances_{random_state}.json")
    with open(download_path, "r") as f:
        feature_importances = json.load(f)
    return feature_importances