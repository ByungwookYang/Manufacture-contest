import sqlite3
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

main_db = "MAIN.DB"

# MAIN.DB안에 train.csv, test.csv 파일을 -> train_table, test_table 로 생성
def csv2db():
    train_df = pd.read_csv("a_data/train.csv")
    test_df = pd.read_csv("a_data/test.csv")

    train_conn = sqlite3.connect(main_db)
    test_conn = sqlite3.connect(main_db)

    train_df.to_sql("train_table", train_conn, if_exists = "replace", index=False)
    test_df.to_sql("test_table", test_conn, if_exists = "replace", index=False)

    print("csv to db")
    train_conn.commit()
    test_conn.commit()
    train_conn.close()
    test_conn.close()

# MAIN 데이터베이스에 저장되어있는 데이터 -> 데이터프레임으로 가져오기(train_table, test_table 등)
def making_dataframe_main_db(table_name):
    con = sqlite3.connect('./MAIN.db')
    cursor = con.cursor()

    cursor.execute(f"SELECT * from {table_name}")
    cols = [column[0] for column in cursor.description]
    dat = pd.DataFrame.from_records(data=cursor.fetchall(), columns=cols)
    
    # 불필요한 열 제거 (errors='ignore'로 열이 없을 경우 무시)
    dat = dat.drop(columns=['rowid', 'Unnamed: 0'], errors='ignore')
    
    # 데이터베이스 연결 종료
    con.commit()
    con.close()
    return dat

# 전처리가 끝난 데이터프레임 -> MAIN.DB에 넣어두기
def create_new_table(data_frame, data_table_name):
    con = sqlite3.connect('./MAIN.db')
    cursor = con.cursor()
    
    # Drop the table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {data_table_name}")
    print(f"\n\nExisting table '{data_table_name}' dropped, and new table is created")
    
    # Check if 'rowid' column exists and remove it
    if 'rowid' in data_frame.columns:
        data_frame = data_frame.drop(columns=['rowid'])
    
    # Create a new table and insert data from the DataFrame
    data_frame.to_sql(data_table_name, con, index=False, if_exists='replace')
    
    # Commit the transaction and close the connection
    con.commit()
    con.close()

## 새로 들어온 데이터를 train데이터로 업데이트(재학습을 위함)
def update_main_db(table_name, new_data):
    try:
        # 데이터베이스에 연결
        conn = sqlite3.connect('./MAIN.db' )
        
        # DataFrame을 데이터베이스에 저장 (기존 테이블 대체)
        new_data.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"테이블 '{table_name}'이(가) 메인 데이터베이스에 성공적으로 업데이트되었습니다.")
    except Exception as e:
        print(f"데이터베이스 업데이트 중 오류 발생: {e}")
    finally:
        # 데이터베이스 연결 종료
        conn.close()

## 새로 들어온 데이터를 train데이터로 업데이트(재학습을 위함)
def update_main_db(table_name, new_data):
    try:
        # 데이터베이스에 연결
        conn = sqlite3.connect('./MAIN.db' )
        
        # DataFrame을 데이터베이스에 저장 (기존 테이블 대체)
        new_data.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"테이블 '{table_name}'이(가) 메인 데이터베이스에 성공적으로 업데이트되었습니다.")
    except Exception as e:
        print(f"데이터베이스 업데이트 중 오류 발생: {e}")
    finally:
        # 데이터베이스 연결 종료
        conn.close()