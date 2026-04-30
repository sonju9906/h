import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import sqlite3
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 (HTML 프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 전역 리소스 로드
model = tf.keras.models.load_model("hairmatch_face_model.keras")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
# train.py의 class_names와 동일한 순서여야 함
class_names = ['Heart Face', 'Long Face', 'Oval Face', 'Round Face'] 

def get_db_recommendation(face_shape):
    conn = sqlite3.connect('capstone_design.db')
    cur = conn.cursor()
    cur.execute("SELECT style_name, advice FROM hair_recommend WHERE face_shape = ?", (face_shape,))
    result = cur.fetchone()
    conn.close()
    return result

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
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
    score = tf.nn.softmax(predictions[0]) # train.py에 softmax가 없다면 필요
    res_shape = class_names[np.argmax(score)]

    # 5. DB 매칭
    recommend = get_db_recommendation(res_shape)

    if recommend:
        return {
            "status": "success",
            "face_shape": res_shape,
            "recommendation": {
                "hair_style": recommend[0],
                "advice": recommend[1]
            }
        }
    return {"status": "error", "message": "추천 정보를 찾을 수 없습니다."}