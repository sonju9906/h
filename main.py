import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import psycopg2  # [수정] sqlite3 대신 PostgreSQL 전용 드라이버 라이브러리 임포트
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine  # 상단에 engine 생성을 위한 모듈 추가

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

# 1. 전역 리소스 로드
model = tf.keras.models.load_model("hairmatch_face_model.keras")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
class_names = ['Heart Face', 'Long Face', 'Oval Face', 'Round Face', 'Square Face']


def get_db_recommendation(face_shape: str, gender: str):
    """face_shape + gender 조합으로 추천 정보 조회 (PostgreSQL 문법 반영)"""
    try:
        # [수정] psycopg2를 활용해 원격 데이터베이스 접속 자원을 획득합니다.
        conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
        cur = conn.cursor()
        
        # [수정] PostgreSQL은 플레이스홀더로 물음표(?) 대신 %s 를 사용해야 합니다.
        cur.execute(
            "SELECT style_name, advice FROM hair_recommend WHERE face_shape = %s AND gender = %s",
            (face_shape, gender)
        )
        result = cur.fetchone()
        return result
    except Exception as e:
        print(f"추천 정보 조회 중 서버 에러 발생: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()


@app.post("/analyze")
async def analyze_face(
    file: UploadFile = File(...),
    gender: str = Form(...)   # ⭐ 프론트에서 보낸 성별 받기
):
    # 0. 성별 값 검증 (DB와 동일하게 'male' / 'female'만 허용)
    if gender not in ("male", "female"):
        raise HTTPException(status_code=400, detail="gender 값은 'male' 또는 'female'이어야 합니다.")

    # 1. 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="올바른 이미지 파일이 아닙니다.")

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. MediaPipe 얼굴 검출
    results = mp_face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return {"status": "error", "message": "얼굴을 찾을 수 없습니다."}

    # 3. 얼굴 영역 크롭 및 전처리
    landmarks = results.multi_face_landmarks[0]
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]

    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

    face_crop = image[max(0, y_min):min(h, y_max), max(0, x_min):min(w, x_max)]
    face_resized = cv2.resize(face_crop, (180, 180))

    img_array = tf.keras.utils.img_to_array(face_resized) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    # 4. 모델 예측
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    res_shape = class_names[np.argmax(score)]

    # 5. DB 매칭 (성별 + 얼굴형)
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
    return {"status": "error", "message": "추천 정보를 찾을 수 없습니다."}