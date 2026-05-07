import tensorflow as tf
import numpy as np
from PIL import Image
import io
import sqlite3
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 (HTML 프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 모델 로드
try:
    model = tf.keras.models.load_model('hairmatch_face_model.keras')
    print("✅ AI 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# ⚠️ 라벨은 train.py의 class_names 순서와 정확히 일치해야 합니다.
# image_dataset_from_directory는 폴더명을 알파벳순으로 자동 정렬하므로 보통 아래 순서가 맞습니다.
# 또한 db_setup.py의 face_shape 값과도 동일해야 DB 조회가 됩니다.
labels = [
    '하트형(Heart Face)',
    '긴형(Long Face)',
    '계란형(Oval Face)',
    '둥근형(Round Face)',
    '사각형(Square Face)',
]


def get_db_recommendation(face_shape: str, gender: str):
    """face_shape + gender 조합으로 추천 정보 조회"""
    conn = sqlite3.connect('capstone_design.db')
    cur = conn.cursor()
    cur.execute(
        "SELECT style_name, advice FROM hair_recommend WHERE face_shape = ? AND gender = ?",
        (face_shape, gender)
    )
    result = cur.fetchone()
    conn.close()
    return result


@app.post("/analyze")
async def analyze_face(
    file: UploadFile = File(...),
    gender: str = Form(...)   # ⭐ 프론트에서 보낸 성별 받기
):
    # 0. 성별 값 검증
    if gender not in ("male", "female"):
        raise HTTPException(status_code=400, detail="gender 값은 'male' 또는 'female'이어야 합니다.")

    # 1. 이미지 읽기
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    # 2. 전처리 (학습 시 사이즈 180x180)
    img = img.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. 예측
    predictions = model.predict(img_array)
    result_idx = int(np.argmax(predictions[0]))
    res_shape = labels[result_idx]

    # 4. DB 매칭 (얼굴형 + 성별)
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
    return {
        "status": "error",
        "message": f"{gender} / {res_shape}에 해당하는 추천 정보를 찾을 수 없습니다."
    }