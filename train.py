import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# --- [1. 설정 단계] ---
img_height = 180
img_width = 180
batch_size = 32
epochs = 20  # 정확도를 위해 조금 더 늘림
dataset_path = "dataset" # 얼굴형별 폴더가 있는 경로

# --- [2. 데이터셋 로드] ---
if not os.path.exists(dataset_path):
    print(f"❌ 에러: '{dataset_path}' 폴더를 찾을 수 없습니다.")
else:
    # 훈련 데이터셋
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # 검증 데이터셋
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ 발견된 클래스: {class_names}")

    # 성능 최적화를 위한 버퍼링 설정
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # --- [3. 모델 구조 정의] ---
    # 데이터 증강 (다양한 각도 대비)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
        # 입력 레이어 및 증강
        layers.Input(shape=(img_height, img_width, 3)),
        data_augmentation,
        
        # 전처리: 0~255 픽셀 값을 0~1 사이로 정규화
        layers.Rescaling(1./255),
        
        # 특징 추출 레이어 (CNN)
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Dropout(0.2), # 과적합 방지
        
        # 분류 레이어
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes) # Logits 출력 (softmax는 main.py에서 처리하거나 여기서 추가)
    ])

    # --- [4. 모델 컴파일] ---
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    # --- [5. 학습 시작] ---
    print(f"\n🚀 {epochs} 에포크 학습을 시작합니다...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # --- [6. 모델 저장] ---
    # .h5보다 최신 방식인 .keras 권장
    model.save("hairmatch_face_model.keras")
    print("\n✅ 모델 저장 완료: hairmatch_face_model.keras")

    # --- [7. 학습 결과 시각화] ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()