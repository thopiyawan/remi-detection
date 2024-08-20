from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import gdown
import os

app = Flask(__name__)
# Google Drive URL สำหรับไฟล์โมเดล
url = 'https://drive.google.com/uc?id=1d1w2HzWzYvIBNPvZSeSBlzRy7rNZQMzQ'
# ชื่อไฟล์โมเดล
output = 'model_inceptionV3.h5'
# ตรวจสอบว่าไฟล์โมเดลมีอยู่แล้วหรือไม่ ถ้าไม่มีก็ดาวน์โหลดจาก Google Drive
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# โหลดโมเดลจากไฟล์ที่ดาวน์โหลดมา
model = load_model(output)

def preprocess_image(img):
    img = img.resize((224, 224))  # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # เพิ่มมิติสำหรับ batch
    img = img / 255.0  # Normalization
    return img

@app.route('/predict', methods=['POST','GET'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image(img)
        
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return jsonify({'prediction': int(predicted_class)})
    
    return jsonify({'error': 'Unable to process the image'}), 400

if __name__ == '__main__':
    # app.run(debug=True, port=5000)
     app.run()
