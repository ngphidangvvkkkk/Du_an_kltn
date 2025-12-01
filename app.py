import os
import cv2
import unicodedata
import tkinter as tk
import tkinter.ttk as ttk
import pandas as pd
import numpy as np

from tkinter import simpledialog, messagebox, filedialog
from datetime import datetime
from PIL import Image

from insightface.app import FaceAnalysis  # ArcFace / InsightFace

# =============== CẤU HÌNH ===============

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LOG_FILE = os.path.join(BASE_DIR, "attendance_log.csv")

os.makedirs(DATASET_DIR, exist_ok=True)

# NGƯỠNG NHẬN DIỆN (ArcFace)
THRESHOLD = 0.65   # cos similarity tối thiểu (có thể chỉnh 0.6–0.7)
VOTE_LIMIT = 3     # số frame đồng ý liên tiếp

# Khởi tạo ArcFace (InsightFace)
# ctx_id = -1: chạy CPU (an toàn cho máy không có GPU)
app_face = FaceAnalysis(name="buffalo_l")
app_face.prepare(ctx_id=-1, det_size=(640, 640))


# =============== HÀM PHỤ TRỢ ===============

def slugify(name: str) -> str:
    """Đổi tên có dấu -> tên thư mục an toàn (không dấu, dùng _)"""
    name = name.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", name)
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))

    cleaned = []
    for c in no_accent:
        if c.isalnum():
            cleaned.append(c)
        elif c in (" ", "-", "_"):
            cleaned.append("_")
    result = "_".join(part for part in "".join(cleaned).split("_") if part)
    return result if result else "person"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def is_blurry(gray_face: np.ndarray, thresh: float = 60.0) -> bool:
    """Kiểm tra ảnh mờ bằng Laplacian variance."""
    lap = cv2.Laplacian(gray_face, cv2.CV_64F)
    return lap.var() < thresh


def read_log_df() -> pd.DataFrame:
    """Đọc file log CSV thành DataFrame, đảm bảo đủ 4 cột."""
    if os.path.isfile(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, encoding="utf-8")
        except Exception:
            df = pd.DataFrame(columns=["date", "name", "check_in", "check_out"])
    else:
        df = pd.DataFrame(columns=["date", "name", "check_in", "check_out"])
    for col in ["date", "name", "check_in", "check_out"]:
        if col not in df.columns:
            df[col] = ""
    df = df[["date", "name", "check_in", "check_out"]]
    return df


def write_log_df(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False, encoding="utf-8")


# =============== ARC FACE: BẮT MẶT + EMBEDDING ===============

def get_face_and_embedding(img_bgr):
    """
    Dùng InsightFace để detect mặt + lấy embedding.
    Trả về (bbox, embedding) với bbox dạng (x1, y1, x2, y2) int,
    hoặc (None, None) nếu không có mặt.
    """
    faces = app_face.get(img_bgr)
    if not faces:
        return None, None

    # Chọn mặt lớn nhất (phòng trường hợp nhiều người)
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    face = faces_sorted[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    emb = l2_normalize(face.embedding.astype("float32"))
    return (x1, y1, x2, y2), emb


# =============== LOAD EMBEDDING ĐÃ LƯU (NHANH) ===============

def load_known_embeddings() -> dict:
    """
    Đọc embedding của từng người trong dataset.
    Mỗi thư mục dataset/<ten_nguoi>/ sẽ có file embedding.npy (trung bình).
    Nếu chưa có embedding.npy thì tính từ ảnh rồi cache lại, dùng cho lần sau.
    """
    known = {}
    if not os.path.exists(DATASET_DIR):
        return known

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        emb_path = os.path.join(person_path, "embedding.npy")
        if os.path.isfile(emb_path):
            emb = np.load(emb_path)
            known[person_name] = l2_normalize(emb.astype("float32"))
            continue

        # Nếu chưa có embedding.npy -> tính lần đầu (có thể hơi lâu),
        # nhưng chỉ 1 lần duy nhất, sau đó cache.
        emb_list = []
        for f in os.listdir(person_path):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_path, f)
            img = cv2.imread(img_path)
            if img is None:
                continue
            _, emb = get_face_and_embedding(img)
            if emb is not None:
                emb_list.append(emb)

        if emb_list:
            mean_emb = np.mean(emb_list, axis=0)
            mean_emb = l2_normalize(mean_emb)
            np.save(emb_path, mean_emb)  # cache lại
            known[person_name] = mean_emb

    # Debug: in ra xem load được mấy người
    print("Loaded identities:", list(known.keys()))
    return known


# =============== GHI LOG CHECK-IN / OUT ===============

def log_attendance_checkin_out(name_folder: str, mode: str) -> bool:
    """
    Ghi log chấm công:
    - date, name, check_in, check_out
    Trả về True nếu có thay đổi, False nếu đã chấm rồi.
    """
    df = read_log_df()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    mask = (df["date"] == today) & (df["name"] == name_folder)
    if not mask.any():
        if mode == "IN":
            new_row = {"date": today, "name": name_folder,
                       "check_in": now_time, "check_out": ""}
        else:
            new_row = {"date": today, "name": name_folder,
                       "check_in": "", "check_out": now_time}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        write_log_df(df)
        return True
    else:
        idx = df.index[mask][0]
        changed = False
        if mode == "IN" and (pd.isna(df.at[idx, "check_in"]) or df.at[idx, "check_in"] == ""):
            df.at[idx, "check_in"] = now_time
            changed = True
        elif mode == "OUT" and (pd.isna(df.at[idx, "check_out"]) or df.at[idx, "check_out"] == ""):
            df.at[idx, "check_out"] = now_time
            changed = True

        if changed:
            write_log_df(df)
        return changed


# =============== TẠO DATASET (LƯU 100 ẢNH + EMBEDDING.NPY) ===============

def capture_dataset(person_name: str):
    """
    Tạo dataset: lưu tối đa 100 ảnh khuôn mặt cho mỗi người
    vào dataset/<slug_ten_nguoi>/, đồng thời tính embedding trung bình
    và lưu vào embedding.npy để check-in dùng cho nhanh.
    """
    folder_name = slugify(person_name)
    save_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    # Giới hạn độ phân giải cho đỡ lag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count, max_images = 0, 100
    emb_list = []
    frame_idx = 0  # xử lý cách frame cho đỡ nặng

    messagebox.showinfo(
        "Thông báo",
        f"Tạo dataset cho {person_name}\nNhấn Q để dừng sớm."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # bỏ qua 1 số frame cho đỡ nặng (vd: chỉ xử lý 1/2 số frame)
        if frame_idx % 2 != 0:
            cv2.imshow("Capturing Dataset (ArcFace)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        img_bgr = frame.copy()
        bbox, emb = get_face_and_embedding(img_bgr)
        if bbox is not None and emb is not None:
            x1, y1, x2, y2 = bbox

            # bỏ mặt quá nhỏ
            if (x2 - x1) < 80 or (y2 - y1) < 80:
                cv2.putText(frame, "Face too small", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                face_crop = img_bgr[y1:y2, x1:x2]
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                if not is_blurry(gray):
                    # lưu ảnh khuôn mặt 160x160
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).resize((160, 160))
                    face_np = cv2.cvtColor(np.asarray(face_pil), cv2.COLOR_RGB2BGR)
                    img_path = os.path.join(save_path, f"img_{count}.jpg")
                    cv2.imwrite(img_path, face_np)

                    emb_list.append(emb)
                    count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Saved:{count}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Capturing Dataset (ArcFace)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Lưu embedding trung bình
    if emb_list:
        mean_emb = np.mean(emb_list, axis=0)
        mean_emb = l2_normalize(mean_emb)
        emb_path = os.path.join(save_path, "embedding.npy")
        np.save(emb_path, mean_emb)

    messagebox.showinfo("Hoàn tất", f"Đã lưu {count} ảnh cho {person_name}")


def add_new_person():
    name = simpledialog.askstring("Tên người mới", "Nhập tên nhân viên:")
    if name and name.strip():
        capture_dataset(name.strip())
    else:
        messagebox.showerror("Lỗi", "Tên không hợp lệ!")


# =============== XEM LỊCH SỬ CÁ NHÂN ===============

def view_history():
    df = read_log_df()
    if df.empty:
        messagebox.showinfo("Thông báo", "Chưa có dữ liệu chấm công.")
        return

    name_input = simpledialog.askstring("Xem lịch sử", "Nhập tên nhân viên:")
    if not name_input or not name_input.strip():
        return

    name_folder = slugify(name_input.strip())
    sub = df[df["name"] == name_folder].copy()

    if sub.empty:
        messagebox.showinfo("Kết quả", f"Không có lịch sử của {name_folder}.")
        return

    sub = sub.sort_values(by="date")

    win = tk.Toplevel(root)
    win.title(f"Lịch sử - {name_folder}")
    win.geometry("480x320")

    tk.Label(win, text=f"Lịch sử: {name_folder}",
             font=("Arial", 11, "bold")).pack(pady=5)

    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    cols = ("date", "check_in", "check_out")
    tree = ttk.Treeview(frame, columns=cols, show="headings")

    tree.heading("date", text="Ngày (YYYY-MM-DD)")
    tree.heading("check_in", text="Check-in")
    tree.heading("check_out", text="Check-out")
    tree.column("date", width=140, anchor="center")
    tree.column("check_in", width=120, anchor="center")
    tree.column("check_out", width=120, anchor="center")

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    for _, r in sub.iterrows():
        tree.insert("", "end", values=(
            r["date"], r["check_in"] or "-", r["check_out"] or "-"
        ))


# =============== XUẤT EXCEL ===============

def export_to_excel():
    df = read_log_df()
    if df.empty:
        messagebox.showinfo("Thông báo", "Chưa có dữ liệu.")
        return

    df = df.sort_values(by=["date", "name"])

    filepath = filedialog.asksaveasfilename(
        title="Chọn nơi lưu file Excel (tất cả dữ liệu)",
        defaultextension=".xlsx",
        filetypes=[("Excel file", "*.xlsx")],
        initialdir=BASE_DIR,
        initialfile="attendance_all.xlsx"
    )
    if not filepath:
        return

    try:
        df.to_excel(filepath, index=False)
        messagebox.showinfo("Thành công", f"Đã xuất file:\n{filepath}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Xuất file thất bại:\n{e}")


def export_filtered_excel():
    df = read_log_df()
    if df.empty:
        messagebox.showinfo("Thông báo", "Chưa có dữ liệu.")
        return

    # lọc ngày
    date_str = simpledialog.askstring(
        "Lọc ngày",
        "Nhập ngày (YYYY-MM-DD hoặc DD/MM/YYYY).\n"
        "Gõ 'all' hoặc để trống để không lọc theo ngày:"
    )
    if date_str:
        date_str = date_str.strip()
        if date_str.lower() != "all" and date_str != "":
            try:
                if "-" in date_str:
                    d = datetime.strptime(date_str, "%Y-%m-%d")
                elif "/" in date_str:
                    d = datetime.strptime(date_str, "%d/%m/%Y")
                else:
                    raise ValueError
                date_key = d.strftime("%Y-%m-%d")
            except Exception:
                messagebox.showerror("Sai định dạng",
                                     "Ngày phải dạng YYYY-MM-DD hoặc DD/MM/YYYY.")
                return
            df = df[df["date"] == date_key]

    # lọc tên
    name_input = simpledialog.askstring(
        "Lọc tên",
        "Nhập tên nhân viên (vd: Phi Đằng).\n"
        "Gõ 'all' hoặc để trống để không lọc theo tên:"
    )
    if name_input:
        name_input = name_input.strip()
        if name_input.lower() != "all" and name_input != "":
            name_folder = slugify(name_input)
            df = df[df["name"] == name_folder]

    if df.empty:
        messagebox.showinfo("Thông báo", "Không có dữ liệu phù hợp.")
        return

    df = df.sort_values(by=["date", "name"])

    filepath = filedialog.asksaveasfilename(
        title="Chọn nơi lưu Excel (lọc theo ngày/tên)",
        defaultextension=".xlsx",
        filetypes=[("Excel file", "*.xlsx")],
        initialdir=BASE_DIR,
        initialfile="attendance_filtered.xlsx"
    )
    if not filepath:
        return

    try:
        df.to_excel(filepath, index=False)
        messagebox.showinfo("Xuất Excel", f"Đã lưu: {filepath}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Xuất thất bại:\n{e}")


# =============== NHẬN DIỆN & CHẤM CÔNG (ARC FACE) ===============

def start_attendance(mode: str):
    """
    Bật camera để check-in / check-out.
    Dùng ArcFace (InsightFace) để nhận diện.
    """
    known = load_known_embeddings()
    if not known:
        messagebox.showwarning("Lỗi", "Chưa có dataset (chưa có embedding nào)!")
        return

    title = "Check-in (Vào)" if mode == "IN" else "Check-out (Ra)"
    messagebox.showinfo(title, f"Bắt đầu {title.lower()}.\nNhấn Q để thoát.")

    cap = cv2.VideoCapture(0)
    # Giảm lag: set độ phân giải vừa phải
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logged_this_session = set()
    votes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_bgr = frame.copy()
        faces = app_face.get(img_bgr)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = l2_normalize(face.embedding.astype("float32"))

            # bỏ mặt quá nhỏ
            if (x2 - x1) < 80 or (y2 - y1) < 80:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Too small", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            # Tìm người có similarity cao nhất
            best_name = "Unknown"
            best = -1.0
            for person, emb_known in known.items():
                sim = cosine_similarity(emb, emb_known)
                if sim > best:
                    best = sim
                    best_name = person

            # print(f"[DEBUG] best={best:.3f}, name={best_name}")

            if best >= THRESHOLD:
                color = (0, 255, 0)
                label = f"{best_name} ({best:.2f})"

                votes.setdefault(best_name, 0)
                votes[best_name] += 1

                if votes[best_name] >= VOTE_LIMIT and best_name not in logged_this_session:
                    updated = log_attendance_checkin_out(best_name, mode)
                    logged_this_session.add(best_name)

                    if updated:
                        msg = "CHECK-IN thành công!" if mode == "IN" else "CHECK-OUT thành công!"
                    else:
                        msg = "Hôm nay đã chấm công rồi."
                    messagebox.showinfo("Thông báo", f"{best_name}: {msg}")
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(title + " (ArcFace)", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Kết thúc", f"Đã dừng {title.lower()}")


# =============== GIAO DIỆN ===============

root = tk.Tk()
root.title("Face Attendance System (ArcFace)")
root.geometry("460x430")

btn_add = btn_history = btn_export_all = btn_export_filter = None


def admin_login():
    login = tk.Toplevel(root)
    login.title("Đăng nhập Admin")
    login.geometry("300x180")
    login.grab_set()

    tk.Label(login, text="Tài khoản:").pack(pady=(15, 2))
    entry_user = tk.Entry(login)
    entry_user.pack(padx=20, fill="x")

    tk.Label(login, text="Mật khẩu:").pack(pady=(10, 2))
    entry_pass = tk.Entry(login, show="*")
    entry_pass.pack(padx=20, fill="x")

    def do_login():
        u = entry_user.get().strip()
        p = entry_pass.get().strip()
        if u == ADMIN_USERNAME and p == ADMIN_PASSWORD:
            messagebox.showinfo("Thành công", "Đăng nhập admin thành công!")
            btn_add.config(state=tk.NORMAL)
            btn_history.config(state=tk.NORMAL)
            btn_export_all.config(state=tk.NORMAL)
            btn_export_filter.config(state=tk.NORMAL)
            login.destroy()
        else:
            messagebox.showerror("Sai thông tin", "Sai tài khoản hoặc mật khẩu.")

    frame_btn = tk.Frame(login)
    frame_btn.pack(pady=15)

    tk.Button(frame_btn, text="Đăng nhập", width=10,
              command=do_login).grid(row=0, column=0, padx=5)
    tk.Button(frame_btn, text="Hủy", width=10,
              command=login.destroy).grid(row=0, column=1, padx=5)

    entry_user.insert(0, ADMIN_USERNAME)
    entry_pass.focus()
    login.bind("<Return>", lambda e: do_login())


# --- Nhân viên ---
tk.Label(root, text="Chức năng cho nhân viên",
         font=("Arial", 11, "bold")).pack(pady=(10, 5))

tk.Button(root, text="Check-in (Vào)",
          command=lambda: start_attendance("IN"),
          width=30, height=2).pack(pady=4)

tk.Button(root, text="Check-out (Ra)",
          command=lambda: start_attendance("OUT"),
          width=30, height=2).pack(pady=4)

# --- Admin ---
tk.Label(root, text="Khu vực Admin",
         font=("Arial", 11, "bold")).pack(pady=(15, 5))

tk.Button(root, text="Đăng nhập Admin",
          command=admin_login,
          width=30, height=2).pack(pady=4)

btn_add = tk.Button(root, text="Thêm người mới (Dataset)",
                    command=add_new_person,
                    width=30, height=2, state=tk.DISABLED)
btn_add.pack(pady=4)

btn_history = tk.Button(root, text="Xem lịch sử chấm công",
                        command=view_history,
                        width=30, height=2, state=tk.DISABLED)
btn_history.pack(pady=4)

btn_export_all = tk.Button(root, text="Xuất Excel (tất cả dữ liệu)",
                           command=export_to_excel,
                           width=30, height=2, state=tk.DISABLED)
btn_export_all.pack(pady=4)

btn_export_filter = tk.Button(root, text="Xuất Excel (lọc ngày/tên)",
                              command=export_filtered_excel,
                              width=30, height=2, state=tk.DISABLED)
btn_export_filter.pack(pady=4)

root.mainloop()
