import sqlite3
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.cluster import KMeans

FEATURE_DB = 'b_analysis/feature_store.db'
# feature engineering

# 1. NULL이 절단 이상인 컬럼 제거 Class
class DropColumnsWithHighNulls(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        self.columns_to_drop = [col for col in X.columns if X[col].isna().sum() / len(X) > self.threshold]
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)

# 2. 1개의 row에서 NULL이 10개 이상이면 해당 행 제거 Class
class DropRowsWithHighNulls(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[X.isnull().sum(axis=1) < self.threshold]

# 3. 컬럼 중 단일값을 갖는 컬럼 제거 Class
class DropConstantColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cols_to_drop = X.columns[X.nunique() <= 1]
        return self

    def transform(self, X):
        return X.drop(self.cols_to_drop, axis=1)

# 4. NULL값 대체하는 Class(molten_temp)
class ClusterImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, temp1_column, temp2_column, n_clusters=3):
        self.n_clusters = n_clusters
        self.target_column = target_column
        self.temp1_column = temp1_column
        self.temp2_column = temp2_column
        self.kmeans = None
        self.cluster_medians = None

    def fit(self, X, y=None):
        if not X[self.target_column].isnull().any():
            return self

        null_dates = X[X[self.target_column].isnull()]['date'].unique()
        X_non_null = X[(X['date'].isin(null_dates)) & (X[self.target_column].notnull())]
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X_non_null[[self.temp1_column, self.temp2_column]])
        
        X_non_null['cluster'] = self.kmeans.predict(X_non_null[[self.temp1_column, self.temp2_column]])
        self.cluster_medians = X_non_null.groupby('cluster')[self.target_column].median()

        return self

    def transform(self, X):
        if not X[self.target_column].isnull().any():
            return X

        null_dates = X[X[self.target_column].isnull()]['date'].unique()
        X_null = X[(X['date'].isin(null_dates)) & (X[self.target_column].isnull())].copy()
        
        X_null['cluster'] = self.kmeans.predict(X_null[[self.temp1_column, self.temp2_column]])
        
        for cluster, median_value in self.cluster_medians.items():
            X_null.loc[X_null['cluster'] == cluster, self.target_column] = median_value
        
        X.update(X_null)
        return X

# 5. NULL값 대체하는 Class(upper_mold_temp3, lower_mold_temp3)
class MoltenTempImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mold_code_medians = None

    def fit(self, X, y=None):
        if not X['molten_temp'].isnull().any():
            return self

        null_dates = X[X['molten_temp'].isnull()]['date'].unique()
        X_non_null = X[(X['date'].isin(null_dates)) & (X['molten_temp'].notnull())]
        self.mold_code_medians = X_non_null.groupby('mold_code')['molten_temp'].median()

        return self

    def transform(self, X):
        if not X['molten_temp'].isnull().any():
            return X

        null_dates = X[X['molten_temp'].isnull()]['date'].unique()
        X_null = X[(X['date'].isin(null_dates)) & (X['molten_temp'].isnull())].copy()

        for mold_code, median_value in self.mold_code_medians.items():
            X_null.loc[X_null['mold_code'] == mold_code, 'molten_temp'] = median_value

        X.update(X_null)
        return X

# 특정 행 제거하는 class
class DropSpecificColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.column_name in X.columns:
            return X.drop(self.column_name, axis=1)
        return X
# NULL값 불러오는 class
class PrintNullInfo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        null_columns = X.columns[X.isna().any()]
        null_counts = X[null_columns].isna().sum()
        print("Columns with Nulls:\n", null_counts)
        return self

    def transform(self, X):
        return X

# 6. 시간을을 시,분 형태로 만들어주는 class
class TimeFormatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='time'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.time_column] = pd.to_datetime(X[self.time_column], errors='coerce')
        X[self.time_column] = X[self.time_column].apply(lambda t: t.strftime('%H:%M') if not pd.isnull(t) else t)
        return X

# 7. 온도 파생변수 생성
class TemperatureFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        """온도 차이와 상/하부 금형 온도 평균을 계산하는 클래스입니다."""
        pass

    def fit(self, X, y=None):
        """파이프라인과 호환을 위해 fit 메서드를 포함합니다."""
        return self

    def transform(self, X):
        """
        온도 차이와 평균 온도 변수를 생성합니다.
        
        Args:
        X (pd.DataFrame): 입력 데이터프레임
        
        Returns:
        pd.DataFrame: 변환된 데이터프레임
        """
        # 온도 차이 변수 생성
        X = X.copy()  # 원본 데이터를 수정하지 않기 위해 복사본 사용
        X['temp_diff1'] = X['upper_mold_temp1'] - X['lower_mold_temp1']
        X['temp_diff2'] = X['upper_mold_temp2'] - X['lower_mold_temp2']
        X['temp_diff3'] = X['upper_mold_temp3'] - X['lower_mold_temp3']
        
        # 상부 금형 온도 평균
        X['upper_mold_avg_temp'] = (
            X['upper_mold_temp1'] + X['upper_mold_temp2'] + X['upper_mold_temp3']
        ) / 3
        
        # 하부 금형 온도 평균
        X['lower_mold_avg_temp'] = (
            X['lower_mold_temp1'] + X['lower_mold_temp2'] + X['lower_mold_temp3']
        ) / 3
        
        return X


# 위의 모든 전처리를 pipeline으로 정의(train)
preprocessing_pipeline_train = Pipeline(steps=[
    ('drop_high_null_columns', DropColumnsWithHighNulls(threshold=0.4)),
    ('drop_high_null_rows', DropRowsWithHighNulls(threshold=10)),
    ('drop_constant_columns', DropConstantColumns()),
    ('time_format_transformer', TimeFormatTransformer(time_column='time')),
    ('upper_imputer', ClusterImputer(target_column='upper_mold_temp3', temp1_column='upper_mold_temp1', temp2_column='upper_mold_temp2', n_clusters=3)),
    ('lower_imputer', ClusterImputer(target_column='lower_mold_temp3', temp1_column='lower_mold_temp1', temp2_column='lower_mold_temp2', n_clusters=3)),
    ('molten_temp_imputer', MoltenTempImputer()),
    ('drop_specific_column', DropSpecificColumn('registration_time')),
    ('temperature_features', TemperatureFeatures())  
])

# 전처리를 pipeline으로 정의(test)
preprocessing_pipeline_test = Pipeline(steps=[
    ('time_format_transformer', TimeFormatTransformer(time_column='time')),
    ('temperature_features', TemperatureFeatures())  
])


# 컬럼 정보만 저장하는 함수
def save_columns_to_feature_store(features, table_name):
    conn = sqlite3.connect(FEATURE_DB)
    
    # 컬럼 정보만 추출
    columns = pd.DataFrame({'column_name': features.columns})
    
    # 컬럼 정보 테이블에 저장
    columns.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    print(f"Column names saved to {table_name} in {FEATURE_DB}")

# test data가 들어왔을 때, 전처리 된 train과 동일한 컬럼만 추출
def load_columns_and_filter_test_data(table_name, test_data):
    conn = sqlite3.connect(FEATURE_DB)
    
    # 컬럼 정보 불러오기
    columns = pd.read_sql(f"SELECT column_name FROM {table_name}", conn)
    conn.close()
    
    # 컬럼 리스트로 변환
    column_list = columns['column_name'].tolist()
    
    # 'passorfail' 컬럼이 있으면 제거
    if 'passorfail' in column_list:
        column_list.remove('passorfail')
    
    # test 데이터에서 해당 컬럼만 남기고 필터링
    filtered_test_data = test_data[column_list]
    return filtered_test_data


def preprocess_data(processed_train, target_column='passorfail'):
    # 수치형 변수 설정
    numerical_cols = [
        'temp_diff1', 'temp_diff2', 'temp_diff3', 'upper_mold_avg_temp', 'lower_mold_avg_temp',
        'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 'low_section_speed', 
        'high_section_speed', 'cast_pressure', 'biscuit_thickness', 'count', 'upper_mold_temp1', 
        'upper_mold_temp2', 'upper_mold_temp3', 'lower_mold_temp1', 'lower_mold_temp2', 
        'lower_mold_temp3', 'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 
        'EMS_operation_time'
    ]
    
    # 범주형 변수 설정
    categorical_cols = list(set(processed_train.columns) - set(numerical_cols) - set([target_column]))
    
    # 범주형 컬럼을 문자열로 변환
    processed_train[categorical_cols] = processed_train[categorical_cols].astype('str')
    # 수치형 컬럼을 float으로 변환
    processed_train[numerical_cols] = processed_train[numerical_cols].astype('float')
    
    # 타겟 변수를 이진 변수로 변환
    processed_train[target_column] = processed_train[target_column].apply(lambda x: 1 if x == 1.0 else 0)
    
    # 독립 변수와 종속 변수 분리
    X = processed_train.drop(target_column, axis=1)
    y = processed_train[target_column]
    
    return X, y, categorical_cols

def preprocess_test(test):
    # 수치형 변수 설정
    numerical_cols = [
        'temp_diff1', 'temp_diff2', 'temp_diff3', 'upper_mold_avg_temp', 'lower_mold_avg_temp',
        'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 'low_section_speed', 
        'high_section_speed', 'cast_pressure', 'biscuit_thickness', 'count', 'upper_mold_temp1', 
        'upper_mold_temp2', 'upper_mold_temp3', 'lower_mold_temp1', 'lower_mold_temp2', 
        'lower_mold_temp3', 'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 
        'EMS_operation_time'
    ]
    
    # 범주형 변수 설정
    categorical_cols = list(set(test.columns) - set(numerical_cols))
    
    # 범주형 컬럼을 문자열로 변환
    test[categorical_cols] = test[categorical_cols].astype('str')
    # 수치형 컬럼을 float으로 변환
    test[numerical_cols] = test[numerical_cols].astype('float')
    
    return test, categorical_cols
