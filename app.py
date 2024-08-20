from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = load_model('my_model.h5')


# Function สำหรับดาวน์โหลดโมเดลจาก Google Drive
def download_model_from_gdrive(model_id, destination):
    url = f"https://drive.google.com/uc?id={model_id}&export=download"
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download model from Google Drive.")

# ใช้ GDrive ID ของโมเดลที่เก็บไว้
model_id = '1d1w2HzWzYvIBNPvZSeSBlzRy7rNZQMzQ'
model_path = '/content/drive/MyDrive/detect-food/model_inceptionV3.h5'  # เก็บไฟล์โมเดลใน temporary directory
download_model_from_gdrive(model_id, model_path)


def preprocess_image(img):
    img = img.resize((224, 224))  # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # เพิ่มมิติสำหรับ batch
    img = img / 255.0  # Normalization
    return img

@app.route('/predict', methods=['POST'])
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
    app.run(debug=True, port=5000)
