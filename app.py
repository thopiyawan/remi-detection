from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = load_model('my_model.h5')

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
