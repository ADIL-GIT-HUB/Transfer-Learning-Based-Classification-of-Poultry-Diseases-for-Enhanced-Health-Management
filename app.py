import streamlit as st
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3
MODEL_PATH = "model/multiclass_model.h5"
DATASET_PATH = "data"

st.set_page_config(page_title="Poultry Disease Classifier", layout="centered", page_icon="🐔")
st.title("🐔 Poultry Disease Classifier (Multi-Class)")
st.markdown("Upload a poultry image to detect **Newcastle Disease**, **Coccidiosis**, **Salmonella**, or **Healthy**.")

# Training function
@st.cache_resource
def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(train.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # LIMIT steps per epoch
    model.fit(
        train,
        validation_data=val,
        epochs=EPOCHS,
        steps_per_epoch=50,
        validation_steps=20,
        verbose=1
    )

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    return model, train.class_indices

# Load or train model
@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        class_labels = sorted(os.listdir(os.path.join(DATASET_PATH, "train")))
        return model, {label: i for i, label in enumerate(class_labels)}
    else:
        return train_model()

# Predict section
model, label_map = load_trained_model()
label_names = {v: k for k, v in label_map.items()}

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(arr)[0]
    top_idx = np.argmax(pred)
    confidence = pred[top_idx] * 100

    st.markdown(f"### 🧠 Prediction: **{label_names[top_idx]}**")
    st.success("Prediction complete. Upload another image if needed.")
