import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Dict, Callable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageFilter

# cấu hình chung
CONFIG = {
    'DATA_PATH': 'Data/train.csv',
    'MODEL_DIR': 'models',
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'MAX_ITER': 500,  # tăng số vòng lặp để hội tụ tốt hơn
    'SOLVER': 'lbfgs' # giải thuật tối ưu cho bài toán nhiều lớp
}

if not os.path.exists(CONFIG['MODEL_DIR']):
    os.makedirs(CONFIG['MODEL_DIR'])

def load_data(path: str):
    try:
        data = pd.read_csv(path)
        Y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
        
        # tự động sửa lỗi thiếu pixel 783
        if X.shape[1] == 783:
            print("tự động thêm pixel thiếu...")
            X = np.hstack((X, np.zeros((X.shape[0], 1))))
            
        return X, Y
    except Exception as e:
        sys.exit(f"lỗi đọc dữ liệu: {e}")

# các hàm xử lý đặc trưng

def feat_1_normalized(X: np.ndarray) -> np.ndarray:
    # chuẩn hóa về khoảng 0-1
    return X.astype(np.float32) / 255.0

def feat_2_edge(X: np.ndarray) -> np.ndarray:
    # dùng pil để phát hiện biên cạnh
    X_new = []
    for i in range(len(X)):
        img_pil = Image.fromarray(X[i].reshape(28, 28).astype(np.uint8))
        edges = img_pil.filter(ImageFilter.FIND_EDGES)
        X_new.append(np.array(edges).flatten())
    return np.array(X_new, dtype=np.float32) / 255.0

def feat_3_block_avg(X: np.ndarray, block_size: int = 4) -> np.ndarray:
    # tính trung bình khối 4x4 để giảm chiều dữ liệu
    N = X.shape[0]
    h, w = 28 // block_size, 28 // block_size
    X_reshaped = X.reshape(N, h, block_size, w, block_size)
    return X_reshaped.mean(axis=(2, 4)).reshape(N, -1)

def feat_4_binarized(X: np.ndarray, threshold: float = 127.5) -> np.ndarray:
    # chuyển về ảnh nhị phân đen trắng
    return (X > threshold).astype(np.float32)

def feat_5_projection(X: np.ndarray) -> np.ndarray:
    # tính tổng pixel theo hàng và cột
    N = X.shape[0]
    X_norm = X.astype(np.float32) / 255.0
    X_proj = np.zeros((N, 56), dtype=np.float32)
    
    for i in range(N):
        img = X_norm[i].reshape(28, 28)
        feat = np.concatenate([img.sum(axis=1), img.sum(axis=0)])
        # chuẩn hóa cục bộ để tránh giá trị lớn
        X_proj[i] = feat / (feat.max() + 1e-12)
        
    return X_proj

if __name__ == "__main__":
    X, Y = load_data(CONFIG['DATA_PATH'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=CONFIG['TEST_SIZE'], random_state=CONFIG['RANDOM_STATE']
    )

    strategies: Dict[str, Callable] = {
        "f1": feat_1_normalized,
        "f2": feat_2_edge,
        "f3": feat_3_block_avg,
        "f4": feat_4_binarized,
        "f5": feat_5_projection
    }

    print("bắt đầu huấn luyện...")

    for name, func in strategies.items():
        print(f"đang xử lý {name}...")
        X_train_proc = func(X_train)
        
        # dùng sklearn thay vì code tự viết để đảm bảo hiệu quả
        model = LogisticRegression(
            multi_class='multinomial', 
            solver=CONFIG['SOLVER'], 
            max_iter=CONFIG['MAX_ITER'],
            random_state=CONFIG['RANDOM_STATE']
        )
        
        model.fit(X_train_proc, Y_train)
        
        # lưu model ra file
        save_path = os.path.join(CONFIG['MODEL_DIR'], f'model_{name}.pkl')
        joblib.dump(model, save_path)
        print(f"đã lưu: {save_path}")

    print("hoàn tất, chạy python app.py để khởi động server")