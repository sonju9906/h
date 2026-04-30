import sqlite3

def setup_database():
    # 1. DB 연결 (파일이 없으면 자동 생성)
    conn = sqlite3.connect('capstone_design.db')
    cur = conn.cursor()

    # 2. 기존 테이블이 있다면 삭제 (초기화용)
    cur.execute('DROP TABLE IF EXISTS hair_recommend')

    # 3. 테이블 생성
    # face_shape: AI 모델의 결과값과 매칭되는 이름
    cur.execute('''
    CREATE TABLE hair_recommend (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_shape TEXT NOT NULL,
        style_name TEXT NOT NULL,
        advice TEXT
    )
    ''')

    # 4. 추천 데이터 리스트 
    # 중요: 첫 번째 항목(face_shape)은 train.py에서 출력된 class_names와 정확히 일치해야 합니다.
    recommend_list = [
        ('Oval Face', '웨이브 단발', '어떤 스타일도 잘 어울리는 얼굴형입니다. 볼륨감 있는 웨이브로 세련미를 더해보세요.'),
        ('Round Face', '보이시한 커트 단발', '얼굴이 길어 보일 수 있도록 정수리 부분에 볼륨을 주고 옆머리는 차분하게 내리는 것을 추천합니다.'),
        ('Square Face', '앞머리 없는 롱헤어', '턱선을 부드럽게 감싸는 레이어드 컷으로 각진 부분을 커버하고 우아한 분위기를 연출하세요.'),
        ('Heart Face', '중단발 C컬 펌', '턱 라인이 뾰족해 보일 수 있으므로 하단에 볼륨을 주는 C컬 스타일로 시선을 분산시키세요.'),
        ('Long Face', '가르마 없는 볼륨 펌', '얼굴이 짧아 보이도록 앞머리를 내리고 옆볼륨을 살린 펌 스타일이 가장 베스트입니다.')
    ]

    # 5. 데이터 삽입
    cur.executemany(
        'INSERT INTO hair_recommend (face_shape, style_name, advice) VALUES (?, ?, ?)', 
        recommend_list
    )

    # 6. 저장 및 종료
    conn.commit()
    conn.close()
    print("✅ DB 설정 완료: 'capstone_design.db' 파일이 생성되었습니다.")

if __name__ == "__main__":
    setup_database()