import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from b_analysis.database import csv2db, making_dataframe_main_db

# 데이터베이스에 데이터를 저장 (한 번 실행해 초기화)
csv2db()

# 데이터 불러오기
train_data = making_dataframe_main_db("train_table")
test_data = making_dataframe_main_db("test_table")

# 1. 데이터의 기본 정보 확인
print("Train Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())

# 2. 결측치 분석
print("Train Data Missing Values:\n", train_data.isnull().sum())
print("\nTest Data Missing Values:\n", test_data.isnull().sum())

# 3. 기본 통계 확인
print("Train Data Statistics:")
print(train_data.describe())

# 수치형으로 변환할 컬럼 리스트
numeric_columns = [
    'molten_temp', 'facility_operation_cycleTime', 'production_cycletime',
    'low_section_speed', 'high_section_speed', 'cast_pressure', 
    'biscuit_thickness', 'count',
    'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
    'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
    'sleeve_temperature', 'physical_strength', 'Coolant_temperature',
    'EMS_operation_time'
]
# 수치형 변수 시각화
def plot_kde_by_passorfail(data, numeric_columns):
    """
    passorfail 변수를 기준으로 각 수치형 변수의 KDE 분포를 그립니다.
    
    Args:
        data (pd.DataFrame): 시각화할 데이터프레임.
        numeric_columns (list): 수치형 변수 컬럼 리스트.
    """
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        
        # passorfail=0 (Fail)과 passorfail=1 (Pass)에 대해 KDE 그리기
        sns.kdeplot(data=data[data['passorfail'] == 0], x=col, label='Fail', fill=True, color='red', alpha=0.5, linewidth=2)
        sns.kdeplot(data=data[data['passorfail'] == 1], x=col, label='Pass', fill=True, color='blue', alpha=0.5, linewidth=2)
        
        plt.title(f'Distribution of {col} by passorfail')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend(title='passorfail')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


