import numpy as np
import joblib
import os
import sys # thêm thư viện sys để dừng chương trình nếu lỗi
from typing import List
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter

# cấu hình server
CONFIG = {
    'MODEL_DIR': 'models',
    'FEATURES': ["f1", "f2", "f3", "f4", "f5"],
    'PORT': 5000,
    'DEBUG': True
}

app = Flask(__name__)
CORS(app)

# biến lưu trữ các model đã load
MODELS = {}

# lấy đường dẫn tuyệt đối của thư mục hiện tại để tránh lỗi path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, CONFIG['MODEL_DIR'])

def load_models():
    # tải các model từ file pkl lên ram
    print(f"loading models from: {MODELS_PATH}...")
    count = 0
    for f in CONFIG['FEATURES']:
        # dùng đường dẫn tuyệt đối để tìm file
        path = os.path.join(MODELS_PATH, f'model_{f}.pkl')
        if os.path.exists(path):
            try:
                MODELS[f] = joblib.load(path)
                count += 1
            except Exception as e:
                print(f"lỗi khi đọc model {f}: {e}")
    
    if count == 0:
        print(f"lỗi nghiêm trọng: không tìm thấy model nào trong {MODELS_PATH}")
        # Không exit ở đây để tránh crash loop trên Render nếu thiếu model, 
        # nhưng API sẽ trả về lỗi 503 khi gọi.
    else:
        print(f"đã load thành công {count} model")

load_models()

def smart_preprocess(pixels: List[float]) -> np.ndarray:
    # chuyển đổi dữ liệu list sang mảng numpy
    arr = np.array(pixels, dtype=np.float32)
    
    # sửa lỗi thiếu pixel 783 thường gặp
    if arr.size == 783: 
        arr = np.append(arr, 0)
    
    img = arr.reshape(28, 28)
    
    # tìm vùng chứa nét vẽ để căn giữa
    rows, cols = np.where(img > 20)
    if len(rows) > 0:
        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)
        crop = img[y_min:y_max+1, x_min:x_max+1]
        
        # tạo ảnh mới nền đen 28x28
        new_img = np.zeros((28, 28))
        h, w = crop.shape
        
        # dán phần nét vẽ vào chính giữa ảnh
        start_y, start_x = (28 - h) // 2, (28 - w) // 2
        new_img[start_y:start_y+h, start_x:start_x+w] = crop
        return new_img
        
    return img

def extract_features(img_28x28: np.ndarray, f_type: str) -> np.ndarray:
    # chuẩn hóa pixel về đoạn 0-1
    norm_img = img_28x28 / 255.0

    if f_type == 'f1':
        # trả về pixel gốc đã chuẩn hóa
        return norm_img.flatten()

    elif f_type == 'f2':
        # dùng bộ lọc tìm biên dạng cạnh
        img_pil = Image.fromarray(img_28x28.astype(np.uint8))
        edges = img_pil.filter(ImageFilter.FIND_EDGES)
        return (np.array(edges, dtype=np.float32) / 255.0).flatten()

    elif f_type == 'f3':
        # tính trung bình theo khối 4x4
        reshaped = img_28x28.reshape(7, 4, 7, 4)
        return reshaped.mean(axis=(1, 3)).flatten()

    elif f_type == 'f4':
        # chuyển về ảnh nhị phân đen trắng
        return (img_28x28 > 127.5).astype(np.float32).flatten()

    elif f_type == 'f5':
        # tính tổng pixel theo hàng và cột
        row_sums = norm_img.sum(axis=1)
        col_sums = norm_img.sum(axis=0)
        feat = np.concatenate([row_sums, col_sums])
        # chuẩn hóa cục bộ để tránh giá trị quá lớn
        return feat / (feat.max() + 1e-12)

    return norm_img.flatten()

# make sure backend do not hibernate on Render if not used for a long time, pinged by Uptimerobot
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "UP", "service": "ML-Flask-Backend"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'pixels' not in data:
            return jsonify({"error": "thiếu dữ liệu pixels"}), 400
            
        pixels = data.get('pixels')
        f_type = data.get('feature_type', 'f1')

        # kiểm tra xem model đã sẵn sàng chưa
        if f_type not in MODELS:
            return jsonify({"error": "model không tồn tại hoặc chưa load xong"}), 503

        # xử lý ảnh đầu vào
        img_centered = smart_preprocess(pixels)
        features = extract_features(img_centered, f_type)
        
        # reshape để đưa vào model sklearn
        final_input = features.reshape(1, -1)

        # thực hiện dự đoán
        model = MODELS[f_type]
        pred_label = int(model.predict(final_input)[0])
        confidence = float(model.predict_proba(final_input).max() * 100)

        return jsonify({
            "result": pred_label,
            "confidence": round(confidence, 2),
            "feature_used": f_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=CONFIG['DEBUG'], port=CONFIG['PORT'])