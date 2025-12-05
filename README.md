# 📸 Hệ thống trích xuất thông tin từ ảnh thẻ sinh viên

> Ứng dụng desktop Python sử dụng OpenCV, VietOCR (Deep Learning) và Face Recognition để tự động trích xuất thông tin từ ảnh thẻ sinh viên, lưu trữ vào MySQL và hỗ trợ tìm kiếm thông minh theo ảnh khuôn mặt với camera realtime.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repository**: [https://github.com/nvn0205/student-card-extractor](https://github.com/nvn0205/student-card-extractor)

---

## ✨ Tính năng chính

### 🔍 Trích xuất thông tin tự động
- ✅ **Tiền xử lý ảnh thông minh**: Tự động chỉnh sáng, tương phản, loại bỏ nhiễu
- ✅ **Phát hiện và cắt thẻ**: Tự động nhận diện vùng thẻ trong ảnh
- ✅ **OCR Deep Learning với VietOCR**: 
  - Sử dụng mô hình Transformer OCR (vgg_seq2seq) được huấn luyện trên 10M+ ảnh tiếng Việt
  - **Tự động tách dòng text**: Phát hiện và OCR từng dòng riêng biệt để tăng độ chính xác
  - Hỗ trợ tiếng Việt đầy đủ với độ chính xác cao
- ✅ **Trích xuất đầy đủ thông tin**:
  - 📝 Mã số sinh viên (MSSV)
  - 👤 Họ và tên (với xử lý OCR noise thông minh)
  - 📅 Ngày sinh
  - 🎓 Niên khóa
  - ⏰ Thẻ có giá trị đến ngày
  - 📸 Ảnh chân dung (tự động cắt với padding động để lấy đủ đầu và cổ)
- ✅ **Loading dialog**: Hiển thị popup "Xin chờ..." khi đang xử lý để người dùng biết hệ thống đang làm việc

### 💾 Quản lý dữ liệu
- ✅ **Lưu trữ MySQL**: Dữ liệu được lưu trữ an toàn với đầy đủ indexes
- ✅ **Tìm kiếm theo tên**: Tìm kiếm nhanh chóng theo tên sinh viên
- ✅ **Tìm kiếm theo khuôn mặt Realtime**: 
  - 📹 **Camera realtime**: Phát hiện và tìm kiếm khuôn mặt trực tiếp từ camera
  - 🤖 **Auto-search**: Tự động tìm kiếm khi phát hiện khuôn mặt
  - 📁 **Upload ảnh**: Tìm kiếm từ file ảnh tải lên
  - ⚡ **Real-time detection**: Phát hiện khuôn mặt trong thời gian thực với khung hiển thị
  - 🎯 **Smart filtering**: Chỉ hiển thị kết quả thực sự khớp (tolerance 0.5, similarity ≥ 60%)
- ✅ **Xem danh sách**: Hiển thị danh sách đầy đủ tất cả sinh viên

### 🎨 Giao diện người dùng
- ✅ **Desktop GUI hiện đại**: Giao diện Tkinter với thiết kế card-based, màu sắc nhất quán
- ✅ **Xem trước thông tin**: Preview thông tin trước khi lưu
- ✅ **Hiển thị ảnh chân dung**: Xem ảnh chân dung được trích xuất với kích thước phù hợp
- ✅ **Chỉnh sửa thủ công**: Có thể chỉnh sửa thông tin sau khi trích xuất
- ✅ **Raw OCR text**: Hiển thị text OCR thô để kiểm tra và debug

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mô tả |
|-----------|-------|
| **Python 3.10+** | Ngôn ngữ lập trình chính |
| **OpenCV** | Xử lý ảnh, phát hiện và tiền xử lý |
| **VietOCR** | Nhận dạng ký tự quang học bằng Deep Learning (Transformer OCR) |
| **PyTorch** | Framework Deep Learning cho VietOCR |
| **face_recognition** | Nhận diện và mã hóa khuôn mặt (dlib-based) |
| **MySQL** | Cơ sở dữ liệu quan hệ |
| **Tkinter** | Giao diện desktop |
| **NumPy** | Xử lý mảng và tính toán |
| **Pillow (PIL)** | Xử lý ảnh Python |

---

## 📋 Yêu cầu hệ thống

### Phần mềm bắt buộc
- **Python**: 3.10 trở lên (khuyến nghị 3.10.x)
- **MySQL Server**: 8.0 trở lên
- **CMake**: Cần thiết cho face-recognition/dlib
- **PyTorch**: Tự động cài khi cài đặt dependencies
- **Webcam/Camera**: Cho tính năng tìm kiếm realtime (tùy chọn, có thể dùng upload ảnh)

### Hệ điều hành hỗ trợ
- ✅ macOS (10.14+)
- ✅ Linux (Ubuntu/Debian)
- ✅ Windows (10/11)

---

## 🚀 Cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/nvn0205/student-card-extractor.git
cd student-card-extractor
```

### Bước 2: Cài đặt Python 3.10

**⚠️ Quan trọng**: Ứng dụng yêu cầu Python 3.10+ để tương thích với VietOCR và các dependencies.

#### Trên macOS:
```bash
# Cài đặt Python 3.10 bằng Homebrew
brew install python@3.10

# Kiểm tra phiên bản
/opt/homebrew/bin/python3.10 -V
```

#### Trên Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3-pip
```

#### Trên Windows:
Tải và cài đặt từ [python.org](https://www.python.org/downloads/) (chọn Python 3.10.x)

### Bước 3: Cài đặt các công cụ cần thiết

#### Trên macOS:
```bash
# Cài đặt CMake (bắt buộc cho face-recognition)
brew install cmake
```

#### Trên Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install cmake
```

#### Trên Windows:
Tải và cài đặt CMake: https://cmake.org/download/

### Bước 4: Cài đặt Python dependencies

**Sử dụng Python 3.10 đã cài ở Bước 2:**

```bash
# macOS (dùng python3.10 từ Homebrew)
/opt/homebrew/bin/python3.10 -m pip install --upgrade pip
/opt/homebrew/bin/python3.10 -m pip install -r requirements.txt

# Ubuntu/Linux (nếu python3.10 là default)
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt

# Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Lưu ý**: 
- Lần đầu cài đặt VietOCR sẽ tự động tải pretrained model (~100MB), có thể mất vài phút
- Nếu gặp lỗi thiếu setuptools hoặc wheel:
  ```bash
  pip install --upgrade setuptools wheel
  pip install -r requirements.txt
  ```

### Bước 5: Cài đặt và cấu hình MySQL

#### Tạo database:
```bash
mysql -u root -p < database/schema.sql
```

Hoặc chạy thủ công trong MySQL:
```sql
CREATE DATABASE IF NOT EXISTS student_card_db;
USE student_card_db;
-- Xem file database/schema.sql để có đầy đủ schema
```

#### Cấu hình kết nối:
Chỉnh sửa file `src/config/database.py`:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'your_password',  # ⚠️ Nhập password MySQL của bạn
    'database': 'student_card_db',
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}
```

---

## 💻 Sử dụng

### Khởi chạy ứng dụng

**Sử dụng Python 3.10 đã cài:**

```bash
# macOS
/opt/homebrew/bin/python3.10 main.py

# Linux/Windows (nếu python3.10 là default)
python3.10 main.py
# hoặc
python main.py
```

### Hướng dẫn sử dụng

#### 1️⃣ Trích xuất thông tin từ thẻ sinh viên

1. Click vào nút **"Trích xuất thông tin"** ở cửa sổ chính
2. Click **"Chọn ảnh thẻ"** và chọn file ảnh thẻ sinh viên
3. Click **"Trích xuất thông tin"** để bắt đầu quá trình xử lý
   - ⏳ Popup "Xin chờ..." sẽ hiển thị trong khi xử lý
4. Xem trước thông tin đã trích xuất:
   - MSSV, Họ tên, Ngày sinh
   - Niên khóa, Ngày hết hạn
   - Ảnh chân dung (hiển thị đầy đủ với kích thước phù hợp)
   - Raw OCR text (để kiểm tra)
5. Kiểm tra và chỉnh sửa thông tin nếu cần
6. Click **"Lưu vào database"** để lưu thông tin

> 💡 **Mẹo**: Ảnh càng rõ nét, độ chính xác OCR càng cao! VietOCR hoạt động tốt nhất với ảnh có độ phân giải cao và đủ ánh sáng.

#### 2️⃣ Tìm kiếm sinh viên theo ảnh khuôn mặt (Camera Realtime)

**Cách 1: Sử dụng Camera Realtime (Khuyến nghị)**
1. Click vào nút **"Tìm kiếm theo ảnh khuôn mặt"**
2. Click **"📹 Bật Camera"** để khởi động camera
3. Đứng trước camera, hệ thống sẽ tự động:
   - Phát hiện khuôn mặt trong thời gian thực
   - Vẽ khung xanh quanh khuôn mặt được phát hiện
   - Tự động tìm kiếm trong database khi phát hiện khuôn mặt (nếu bật auto-search)
4. Xem kết quả tự động hiển thị khi tìm thấy khớp:
   - Chỉ hiển thị kết quả thực sự khớp (độ tương đồng ≥ 60%)
   - Màu xanh nhạt: Khớp chính xác (matched)
   - Màu vàng nhạt: Tương đồng cao (similar)
5. Click vào một sinh viên trong danh sách để xem chi tiết
6. Click **"⏹️ Tắt Camera"** khi hoàn thành

**Tùy chọn:**
- ✅ Bật/tắt **"Tự động tìm kiếm"**: Tự động tìm kiếm khi phát hiện khuôn mặt
- 📁 **"Chọn ảnh"**: Upload ảnh từ file (phương pháp cũ)

**Cách 2: Tìm kiếm từ file ảnh**
1. Click vào nút **"Tìm kiếm theo ảnh khuôn mặt"**
2. Click **"📁 Chọn ảnh"** và chọn ảnh cần tìm
3. Hệ thống sẽ tự động tìm kiếm và hiển thị kết quả
4. Xem danh sách kết quả (sắp xếp theo độ tương đồng)
5. Click vào một sinh viên trong danh sách để xem chi tiết

> 💡 **Mẹo**: Camera realtime hoạt động tốt nhất với ánh sáng đủ và khuôn mặt nhìn thẳng vào camera!

#### 3️⃣ Xem danh sách sinh viên

1. Click vào nút **"Xem danh sách sinh viên"**
2. Xem toàn bộ sinh viên đã được lưu trong database
3. Sắp xếp theo các cột bằng cách click vào header
4. Cuộn để xem thêm

---

## 📁 Cấu trúc dự án

```
BTL/
├── 📄 main.py                    # Entry point (wrapper)
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Tài liệu hướng dẫn
│
├── 📁 config/                    # Cấu hình
│   ├── __init__.py
│   └── database.py              # Cấu hình kết nối MySQL
│
├── 📁 database/                  # Database schemas
│   └── schema.sql               # Schema tạo bảng students
│
├── 📁 src/                       # Source code chính
│   ├── __init__.py
│   ├── main.py                  # Entry point chính
│   │
│   ├── 📁 image_processing/     # Xử lý ảnh
│   │   ├── __init__.py
│   │   ├── preprocessor.py     # Tiền xử lý ảnh
│   │   └── card_detector.py    # Phát hiện và cắt thẻ
│   │
│   ├── 📁 extraction/           # Trích xuất dữ liệu
│   │   ├── __init__.py
│   │   ├── ocr_extractor.py    # Trích xuất text bằng VietOCR (multi-line)
│   │   └── face_extractor.py   # Trích xuất ảnh chân dung
│   │
│   ├── 📁 database/             # Database operations
│   │   ├── __init__.py
│   │   ├── db_manager.py       # Quản lý kết nối DB
│   │   └── student_dao.py      # CRUD operations
│   │
│   ├── 📁 face_matching/        # Face recognition
│   │   ├── __init__.py
│   │   └── face_matcher.py     # So khớp khuôn mặt
│   │
│   └── 📁 gui/                  # Giao diện người dùng
│       ├── __init__.py
│       ├── main_window.py      # Cửa sổ chính
│       ├── extract_window.py   # Cửa sổ trích xuất
│       └── search_window.py    # Cửa sổ tìm kiếm
│
├── 📁 avatars/                  # Thư mục lưu ảnh chân dung (tự động tạo)
│   └── ...
│
└── 📁 tests/                    # Unit tests (nếu có)
    └── ...
```

---

## 🎯 Cơ sở dữ liệu

### Schema bảng `students`

| Cột | Kiểu dữ liệu | Mô tả |
|-----|--------------|-------|
| `id` | INT (PRIMARY KEY) | ID tự động tăng |
| `mssv` | VARCHAR(20) UNIQUE | Mã số sinh viên |
| `ho_ten` | VARCHAR(100) | Họ và tên |
| `ngay_sinh` | DATE | Ngày sinh |
| `nien_khoa` | VARCHAR(20) | Niên khóa (format: YYYY-YYYY) |
| `ngay_het_han` | DATE | Thẻ có giá trị đến ngày |
| `avatar_path` | TEXT | Đường dẫn file ảnh chân dung |
| `face_encoding` | BLOB | Vector mã hóa khuôn mặt (128D) |
| `created_at` | TIMESTAMP | Thời gian tạo record |

### Indexes
- `idx_mssv`: Index trên cột `mssv` (tìm kiếm nhanh)
- `idx_ho_ten`: Index trên cột `ho_ten` (tìm kiếm theo tên)

---

## ⚙️ Các tính năng kỹ thuật

### 🔬 OCR Engine (VietOCR)

- **Deep Learning Model**: 
  - Sử dụng mô hình Transformer OCR (vgg_seq2seq) được huấn luyện trên 10M+ ảnh
  - Pretrained model tự động tải về khi khởi chạy lần đầu
  - Độ chính xác cao với tiếng Việt và chữ số

- **Multi-line Text Detection**:
  - Tự động phát hiện và tách các dòng text trên thẻ
  - Sử dụng threshold + morphological operations (dilation ngang) để gộp ký tự thành dòng
  - OCR từng dòng riêng biệt để tăng độ chính xác
  - Ghép kết quả thành chuỗi multi-line để parser xử lý

- **Smart Text Parsing**:
  - Regex patterns linh hoạt để xử lý OCR noise và typo
  - Xử lý đặc biệt cho các trường hợp OCR sai (ví dụ: `3111/2027` → `31/12/2027`)
  - Ưu tiên lấy substring ngay sau label (ví dụ: "Họ & tên: ...")
  - Fallback patterns để đảm bảo tìm được thông tin ngay cả khi OCR không hoàn hảo

### 🤖 Face Recognition
- **Dual model support**: HOG (nhanh) và CNN (chính xác)
- **Auto resizing**: Tự động resize ảnh nhỏ để tăng độ chính xác
- **Dynamic padding**: Padding động dựa trên kích thước khuôn mặt để lấy đủ đầu và cổ
- **128-dimensional encoding**: Mã hóa khuôn mặt thành vector 128D
- **Distance-based matching**: So khớp dựa trên khoảng cách Euclidean
- **Smart filtering**: Chỉ hiển thị kết quả thực sự khớp (tolerance 0.5, similarity ≥ 60%)
- **Realtime camera support**: 
  - Video streaming từ webcam/camera
  - Face detection overlay với khung xanh
  - Auto-search mỗi giây khi phát hiện khuôn mặt
  - Multi-threading để không block UI

### 🖼️ Image Processing
- **Automatic card detection**: Tự động phát hiện vùng thẻ
- **Noise reduction**: Giảm nhiễu ảnh
- **Contrast enhancement**: Tăng cường độ tương phản
- **Smart cropping**: Cắt chính xác vùng quan tâm
- **Text line detection**: Phát hiện và tách các dòng text

---

## 🐛 Xử lý lỗi thường gặp

### ❌ Lỗi: `CMake is not installed on your system!`
**Nguyên nhân**: Thiếu CMake để build face-recognition/dlib

**Giải pháp**:
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake

# Sau đó cài lại dependencies
pip install -r requirements.txt
```

---

### ❌ Lỗi: `No module named 'torch'` hoặc `No module named 'torchvision'`
**Nguyên nhân**: Thiếu PyTorch (cần cho VietOCR)

**Giải pháp**:
```bash
# Cài đặt PyTorch và torchvision
pip install torch torchvision

# Hoặc cài lại toàn bộ dependencies
pip install -r requirements.txt
```

---

### ❌ Lỗi: `Can't connect to MySQL server`
**Nguyên nhân**: MySQL chưa chạy hoặc thông tin kết nối sai

**Giải pháp**:
1. Kiểm tra MySQL Server đang chạy:
   ```bash
   # macOS/Linux
   mysql -u root -p
   
   # Nếu không kết nối được, khởi động MySQL
   # macOS
   brew services start mysql
   
   # Linux
   sudo systemctl start mysql
   ```

2. Kiểm tra file `src/config/database.py` có đúng thông tin không
3. Đảm bảo database `student_card_db` đã được tạo

---

### ❌ Lỗi: `Unknown column 'lop' in 'field list'`
**Nguyên nhân**: Schema database không khớp với code

**Giải pháp**:
```bash
# Chạy lại script tạo database
mysql -u root -p < database/schema.sql
```

---

### ❌ OCR không nhận diện đúng
**Nguyên nhân**: Ảnh chất lượng kém hoặc độ sáng không đủ

**Giải pháp**:
- Sử dụng ảnh có độ phân giải cao (tối thiểu 800x600)
- Đảm bảo ảnh đủ sáng, rõ nét
- Tránh bóng, phản quang trên thẻ
- Chụp ảnh thẳng góc, không bị nghiêng
- VietOCR hoạt động tốt nhất với ảnh có độ tương phản cao

---

### ❌ Không phát hiện được khuôn mặt
**Nguyên nhân**: Ảnh quá nhỏ hoặc khuôn mặt không rõ

**Giải pháp**:
- Sử dụng ảnh có chất lượng tốt
- Đảm bảo khuôn mặt rõ ràng, không bị che khuất
- Code đã tự động resize ảnh nhỏ và thử cả HOG và CNN model
- Với camera realtime: đảm bảo đủ ánh sáng và nhìn thẳng vào camera

---

### ❌ Lỗi: Camera không mở được
**Nguyên nhân**: Camera bị chiếm dụng hoặc không kết nối

**Giải pháp**:
- Kiểm tra camera đã được kết nối và không bị ứng dụng khác sử dụng
- Trên macOS: Cấp quyền truy cập camera cho Terminal/Python trong System Preferences
- Trên Linux: Đảm bảo user có quyền truy cập `/dev/video0`
- Thử khởi động lại ứng dụng
- Nếu có nhiều camera, có thể cần chỉnh sửa `cv2.VideoCapture(0)` thành index khác (1, 2, ...)

---

### ❌ Lỗi: `macOS 26 (2601) or later required`
**Nguyên nhân**: Đang dùng Python 3.14+ (quá mới, không tương thích với một số dependencies)

**Giải pháp**:
- Cài đặt Python 3.10.x (khuyến nghị 3.10.14)
- Sử dụng đúng Python 3.10 để chạy ứng dụng:
  ```bash
  # macOS
  /opt/homebrew/bin/python3.10 main.py
  
  # Kiểm tra phiên bản
  /opt/homebrew/bin/python3.10 -V
  ```

---

## 📝 Lưu ý quan trọng

- ✅ **Python Version**: Bắt buộc Python 3.10+ (không dùng 3.14+ vì không tương thích)
- ✅ **Ảnh chất lượng**: Ảnh thẻ càng rõ nét, độ chính xác OCR càng cao
- ✅ **VietOCR**: Model sẽ tự động tải về lần đầu chạy (~100MB), có thể mất vài phút
- ✅ **Hiệu năng**: Face recognition sử dụng CNN model có thể chậm hơn, nhưng chính xác hơn
- ✅ **MySQL**: Đảm bảo MySQL đang chạy trước khi sử dụng ứng dụng
- ✅ **Backup**: Nên backup database thường xuyên
- ✅ **Multi-line OCR**: VietOCR tự động tách và OCR từng dòng, giúp tăng độ chính xác đáng kể

---

## 🔄 Quy trình hoạt động chi tiết

### Luồng trích xuất thông tin

1. **Chọn ảnh** → Load ảnh bằng OpenCV, resize để preview
2. **Detect thẻ** → Tự động phát hiện vùng thẻ (nếu chụp cả mặt bàn)
3. **Tách dòng text** → Dùng threshold + dilation để phát hiện các dòng text
4. **OCR từng dòng** → VietOCR nhận diện từng dòng riêng biệt
5. **Ghép kết quả** → Tạo chuỗi multi-line text
6. **Parse thông tin** → Regex patterns để trích xuất MSSV, Họ tên, Ngày sinh, Niên khóa, Ngày hết hạn
7. **Trích xuất ảnh chân dung** → Face detection với padding động
8. **Hiển thị & chỉnh sửa** → Người dùng có thể chỉnh sửa trước khi lưu
9. **Lưu vào DB** → Lưu thông tin + ảnh chân dung + face encoding

### Luồng tìm kiếm khuôn mặt

1. **Bật camera** → Stream video từ webcam
2. **Detect khuôn mặt** → Face detection trong mỗi frame
3. **Encode khuôn mặt** → Tạo 128D vector
4. **So khớp** → Tính distance với tất cả face encodings trong DB
5. **Filter kết quả** → Chỉ hiển thị kết quả khớp (tolerance 0.5, similarity ≥ 60%)
6. **Hiển thị** → Danh sách kết quả với độ tương đồng

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Nếu bạn muốn:

- 🐛 Báo lỗi: Tạo [Issue](https://github.com/nvn0205/student-card-extractor/issues)
- 💡 Đề xuất tính năng: Tạo [Feature Request](https://github.com/nvn0205/student-card-extractor/issues)
- 🔧 Submit PR: Fork repo và tạo [Pull Request](https://github.com/nvn0205/student-card-extractor/pulls)

---

## 📄 License

Dự án này được phát hành dưới giấy phép [MIT License](LICENSE).

---

## 👨‍💻 Tác giả

Được phát triển cho **Bài tập lớn môn Xử lý ảnh**.

---

## 🙏 Lời cảm ơn

- [OpenCV](https://opencv.org/) - Thư viện xử lý ảnh
- [VietOCR](https://github.com/pbcquoc/vietocr) - Công cụ OCR Deep Learning cho tiếng Việt
- [face_recognition](https://github.com/ageitgey/face_recognition) - Thư viện nhận diện khuôn mặt
- [dlib](http://dlib.net/) - Machine learning library
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

**⭐ Nếu dự án hữu ích, hãy cho một [star](https://github.com/nvn0205/student-card-extractor) nhé! ⭐**
