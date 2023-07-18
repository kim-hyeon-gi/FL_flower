import csv
import sqlite3

import pandas as pd
import tqdm

# 데이터베이스 연결
conn = sqlite3.connect(
    "/home/hyeongikim/Desktop/FL/dataset/reddit/database.sqlite"
)  # 데이터베이스 파일 경로를 지정합니다.

# 커서 생성
cursor = conn.cursor()

# 테이블 목록 가져오기
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# 첫 번째 테이블 선택
table_name = tables[0][0]

# id와 body 추출
cursor.execute(f"SELECT id, body FROM {table_name};")
data = cursor.fetchall()

# id로 정렬
data.sort(key=lambda x: x[0])

print("complete sort")
# 텍스트 파일에 저장
file = open("./dataset/reddit/server_data.txt", "w")
client_file = open("./dataset/reddit/client.csv", "w")
writer = csv.writer(client_file)
writer.writerow(["text"])  # 헤더 작성

current_id = "start_id"
current_word_count = 0
current_body = "start"
server_word_count = 0
row_num = 0
for row in data:
    id_value = row[0]
    body_value = row[1]

    if current_id != id_value:
        if 1600 <= current_word_count <= 45000:  # 1600~45000
            writer.writerow([current_body])
            row_num = row_num + 1

        else:
            if server_word_count < 3000000:
                file.write(current_body)
                server_word_count = server_word_count + current_word_count
        current_id = id_value
        current_word_count = 0
        current_body = ""
    word_count = len(body_value.split())
    current_body = current_body + body_value
    current_word_count += word_count
print(row_num)

# 연결 종료
conn.close()
