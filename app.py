import tensorflow as tf
import numpy as np
from PIL import Image
import io
import hashlib
import psycopg2  # [수정] sqlite3 대신 PostgreSQL 전용 드라이버 라이브러리 임포트
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine  # 상단에 누락되어 있던 create_engine 추가

app = FastAPI()

# CORS 설정 (HTML 프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# [수정] B 방식 적용: 변수명 및 URL 지정
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:설치시비밀번호@localhost:5432/hairmatch_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 비밀번호 해싱 함수 (보안용)
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------------------------------------------------
# 1. 회원가입 API (PostgreSQL 문법 반영)
# ---------------------------------------------------------
@app.post("/signup")
async def signup(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    name: str = Form(...),
    gender: str = Form(...)
):
    # [수정] sqlite3.connect 대신 psycopg2를 활용해 데이터베이스 접속 자원을 획득합니다.
    try:
        conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
        cursor = conn.cursor()
        
        hashed_pw = hash_password(password)
        
        # [수정] PostgreSQL은 플레이스홀더로 물음표(?) 대신 %s 를 사용해야 합니다.
        cursor.execute(
            "INSERT INTO users (username, password, email, name, gender) VALUES (%s, %s, %s, %s, %s)",
            (username, hashed_pw, email, name, gender)
        )
        conn.commit()
        return {"status": "success", "message": "회원가입이 완료되었습니다."}
    
    # [수정] SQLite 예외 규격 대신 psycopg2의 무결성 제약 조건 에러를 처리합니다.
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
    
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# ---------------------------------------------------------
# 2. 로그인 API (PostgreSQL 문법 반영)
# ---------------------------------------------------------
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    try:
        conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
        cursor = conn.cursor()
        
        hashed_pw = hash_password(password)
        
        # [수정] 플레이스홀더를 %s 로 변경
        cursor.execute("SELECT name, gender FROM users WHERE username = %s AND password = %s", (username, hashed_pw))
        user = cursor.fetchone()
        return {
            "status": "success", 
            "message": f"{user[0]}님 환영합니다!", 
            "user_name": user[0],
            "user_gender": user[1]
        } if user else HTTPException(status_code=401, detail="아이디 또는 비밀번호가 일치하지 않습니다.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"로그인 처리 중 서버 오류가 발생했습니다: {str(e)}")
        
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# ---------------------------------------------------------
# 3. AI 모델 로드 및 얼굴 분석 API
# ---------------------------------------------------------
try:
    model = tf.keras.models.load_model('hairmatch_face_model.keras')
    print("✅ AI 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

labels = [
    '하트형(Heart Face)',
    '긴형(Long Face)',
    '계란형(Oval Face)',
    '둥근형(Round Face)',
    '사각형(Square Face)',
]

def get_db_recommendation(face_shape: str, gender: str):
    try:
        conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
        cur = conn.cursor()
        
        # [수정] 플레이스홀더를 %s 로 변경
        cur.execute(
            "SELECT style_name, advice FROM hair_recommend WHERE face_shape = %s AND gender = %s",
            (face_shape, gender)
        )
        result = cur.fetchone()
        return result
    except Exception as e:
        print(f"추천 정보 조회 실패: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()

@app.post("/analyze")
async def analyze_face(
    file: UploadFile = File(...),
    gender: str = Form(...)
):
    if gender not in ("male", "female"):
        raise HTTPException(status_code=400, detail="gender 값은 'male' 또는 'female'이어야 합니다.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    # 전처리 (학습 데이터와 동일한 180x180 사이즈)
    img = img.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    predictions = model.predict(img_array)
    result_idx = int(np.argmax(predictions[0]))
    res_shape = labels[result_idx]

    # DB에서 결과 조회 (얼굴형 + 성별)
    recommend = get_db_recommendation(res_shape, gender)

    if recommend:
        return {
            "status": "success",
            "face_shape": res_shape,
            "gender": gender,
            "recommendation": {
                "hair_style": recommend[0],
                "advice": recommend[1]
            }
        }
    
    raise HTTPException(status_code=404, detail=f"{gender} / {res_shape}에 대한 추천 정보가 없습니다.")