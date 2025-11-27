import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import csv
from datetime import datetime

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ================== MÔ HÌNH & THƯ MỤC ==================

# MTCNN detect
detector = MTCNN()

# Model trích xuất đặc trưng khuôn mặt (dùng MobileNetV2 pretrained)
embed_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Tự tạo thư mục dataset
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

LOG_FILE = "attendance_log.csv"


# ================== HÀM EMBEDDING & LOAD DATASET ==================

def get_embedding(face_rgb):
    """
    Nhận ảnh khuôn mặt (RGB) dạng numpy, resize & trích xuất embedding.
    """
    # resize về kích thước đầu vào của MobileNetV2 (224x224)
    img = cv2.resize(face_rgb, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = embed_model.predict(img, verbose=0)[0]
    # chuẩn hoá vector
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def load_known_embeddings():
    """
    Đọc toàn bộ ảnh trong folder dataset/<ten_nguoi>/,
    tính embedding trung bình cho mỗi người.
    Trả về: dict {ten_nguoi: embedding_vector}
    """
    known = {}

    if not os.path.exists(DATASET_DIR):
        return known

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        emb_list = []
        for file_name in os.listdir(person_path):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, file_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            emb = get_embedding(img_rgb)
            emb_list.append(emb)

        if emb_list:
            mean_emb = np.mean(emb_list, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            known[person_name] = mean_emb

    return known


def cosine_similarity(a, b):
    return np.dot(a, b)


# ================== HÀM LOG CHẤM CÔNG ==================

def log_attendance(name, similarity):
    """
    Ghi log chấm công vào file CSV:
    timestamp, name, similarity
    """
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "name", "similarity"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            f"{similarity:.4f}"
        ])


# ================== HÀM CHỤP DATASET (BẠN ĐÃ CÓ) ==================

def capture_dataset(person_name):
    save_path = f"{DATASET_DIR}/{person_name}"
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    max_images = 80  # số ảnh bạn muốn chụp

    messagebox.showinfo("Thông báo",
                        f"Đang tạo dataset cho: {person_name}\nNhấn Q để dừng sớm.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for face in faces:
            x, y, w, h = face["box"]
            x, y = abs(x), abs(y)

            # Chặn tràn biên
            h_img, w_img, _ = rgb.shape
            x2 = min(x + w, w_img)
            y2 = min(y + h, h_img)

            face_img = rgb[y:y2, x:x2]

            if face_img.size == 0:
                continue

            # Lưu mỗi khuôn mặt crop lại
            face_img = Image.fromarray(face_img)
            face_img = face_img.resize((160, 160))
            face_np = np.asarray(face_img)

            img_path = f"{save_path}/img_{count}.jpg"
            cv2.imwrite(img_path, cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))

            count += 1
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {count}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capturing Dataset", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Hoàn tất",
                        f"Tạo dataset thành công cho {person_name}\nTổng ảnh: {count}")


def add_new_person():
    person_name = simpledialog.askstring("Tên người mới", "Nhập tên người cần thêm:")
    if person_name and person_name.strip() != "":
        capture_dataset(person_name.strip())
    else:
        messagebox.showerror("Lỗi", "Tên không hợp lệ!")


# ================== HÀM NHẬN DIỆN & CHẤM CÔNG ==================

def start_attendance():
    # Load embeddings từ dataset
    known_embeddings = load_known_embeddings()
    if not known_embeddings:
        messagebox.showwarning("Chưa có dữ liệu",
                               "Chưa có dataset nào.\nHãy thêm người mới trước.")
        return

    messagebox.showinfo("Chấm công",
                        "Bắt đầu nhận diện chấm công.\nNhấn Q để thoát.")

    cap = cv2.VideoCapture(0)

    # Để tránh ghi log trùng lặp trong cùng 1 buổi,
    # ta chỉ log mỗi người 1 lần cho mỗi lần chạy hàm.
    logged_this_session = set()

    THRESHOLD = 0.7  # ngưỡng giống ≥ 70%

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for face in faces:
            x, y, w, h = face["box"]
            x, y = abs(x), abs(y)

            h_img, w_img, _ = rgb.shape
            x2 = min(x + w, w_img)
            y2 = min(y + h, h_img)
            face_img = rgb[y:y2, x:x2]

            if face_img.size == 0:
                continue

            # Lấy embedding khuôn mặt hiện tại
            emb = get_embedding(face_img)

            # So khớp với tất cả người trong dataset
            best_name = "Unknown"
            best_sim = -1.0

            for person_name, known_emb in known_embeddings.items():
                sim = cosine_similarity(emb, known_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = person_name

            # Vẽ khung & ghi log nếu giống trên 70%
            if best_sim >= THRESHOLD:
                color = (0, 255, 0)  # xanh
                label = f"{best_name} ({best_sim:.2f})"

                # Ghi log nếu chưa log người này trong buổi này
                if best_name not in logged_this_session:
                    log_attendance(best_name, best_sim)
                    logged_this_session.add(best_name)
            else:
                color = (0, 0, 255)  # đỏ
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Kết thúc", "Đã dừng chấm công.")


# ================== TẠO GUI TKINTER ==================

root = tk.Tk()
root.title("Face Attendance & Dataset Creator")
root.geometry("350x220")

btn_attendance = tk.Button(
    root,
    text="Bắt đầu chấm công (Nhận diện)",
    command=start_attendance,
    width=30,
    height=2
)
btn_attendance.pack(pady=10)

btn_add = tk.Button(
    root,
    text="Thêm người mới (Tạo Dataset)",
    command=add_new_person,
    width=30,
    height=2
)
btn_add.pack(pady=10)

root.mainloop()
