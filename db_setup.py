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
        gender TEXT NOT NULL,  -- 'male' 또는 'female'
        face_shape TEXT NOT NULL,
        style_name TEXT NOT NULL,
        advice TEXT
    )
    ''')

    # 4. 추천 데이터 리스트 
    # 중요: 첫 번째 항목(face_shape)은 train.py에서 출력된 class_names와 정확히 일치해야 합니다.
    recommend_list = [
        # --- 여성 (female) ---
        ('female', '계란형(Oval Face)', '웨이브 단발', '세련미를 더하는 볼륨 웨이브를 추천합니다.'),
        ('female', '둥근형(Round Face)', '레이어드 컷', '옆머리로 볼살을 커버하여 갸름해 보이는 효과를 줍니다.'),
        ('female', '사각형(Square Face)', '사이드뱅 롱헤어', '부드러운 곡선 위주로 각진 턱선을 가려보세요.'),
        ('female', '하트형(Heart Face)', '중단발 C컬 펌', '하단 볼륨으로 뾰족한 턱 끝을 보완합니다.'),
        ('female', '긴형(Long Face)', '풀뱅 앞머리', '얼굴 길이를 짧아 보이게 하는 앞머리 스타일이 베스트입니다.'),
        
        # --- 남성 (male) ---
        ('male', '계란형(Oval Face)', '리프 컷', '자연스러운 가르마로 부드러운 이미지를 강조하세요.'),
        ('male', '둥근형(Round Face)', '리젠트 컷', '옆머리는 짧게, 윗머리는 세워 얼굴이 길어 보이게 합니다.'),
        ('male', '사각형(Square Face)', '포마드 스타일', '깔끔한 가르마로 남성적이고 신뢰감 있는 인상을 줍니다.'),
        ('male', '하트형(Heart Face)', '내린 머리 쉐도우 펌', '이마를 가려 턱으로 가는 시선을 분산시킵니다.'),
        ('male', '긴형(Long Face)', '가르마 펌 (사이드 볼륨)', '옆 볼륨을 살려 가로 폭을 넓어 보이게 하세요.')
    ]
    # 5. 데이터 삽입
    cur.executemany(
        'INSERT INTO hair_recommend (gender, face_shape, style_name, advice) VALUES (?, ?, ?, ?)', 
        recommend_list
    )

    # 6. 저장 및 종료
    conn.commit()
    conn.close()
    print("✅ DB 설정 완료: 'capstone_design.db' 파일이 생성되었습니다.")

if __name__ == "__main__":
    setup_database()