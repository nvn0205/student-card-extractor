# 📸 Hệ thống trích xuất thông tin từ ảnh thẻ sinh viên

> Ứng dụng desktop Python sử dụng OpenCV, Tesseract OCR và Face Recognition để tự động trích xuất thông tin từ ảnh thẻ sinh viên, lưu trữ vào MySQL và hỗ trợ tìm kiếm thông minh theo ảnh khuôn mặt.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repository**: [https://github.com/nvn0205/student-card-extractor](https://github.com/nvn0205/student-card-extractor)

---

## ✨ Tính năng chính

### 🔍 Trích xuất thông tin tự động
- ✅ **Tiền xử lý ảnh thông minh**: Tự động chỉnh sáng, tương phản, loại bỏ nhiễu
- ✅ **Phát hiện và cắt thẻ**: Tự động nhận diện vùng thẻ trong ảnh
- ✅ **OCR đa phương thức**: Sử dụng nhiều thuật toán preprocessing và PSM modes để tối ưu độ chính xác
- ✅ **Trích xuất đầy đủ thông tin**:
  - 📝 Mã số sinh viên (MSSV)
  - 👤 Họ và tên
  - 📅 Ngày sinh
  - 🎓 Niên khóa
  - ⏰ Thẻ có giá trị đến ngày
  - 📸 Ảnh chân dung

### 💾 Quản lý dữ liệu
- ✅ **Lưu trữ MySQL**: Dữ liệu được lưu trữ an toàn với đầy đủ indexes
- ✅ **Tìm kiếm theo tên**: Tìm kiếm nhanh chóng theo tên sinh viên
- ✅ **Tìm kiếm theo khuôn mặt**: Sử dụng AI để nhận diện và tìm kiếm theo ảnh khuôn mặt
- ✅ **Xem danh sách**: Hiển thị danh sách đầy đủ tất cả sinh viên

### 🎨 Giao diện người dùng
- ✅ **Desktop GUI thân thiện**: Giao diện Tkinter dễ sử dụng
- ✅ **Xem trước thông tin**: Preview thông tin trước khi lưu
- ✅ **Hiển thị ảnh chân dung**: Xem ảnh chân dung được trích xuất
- ✅ **Chỉnh sửa thủ công**: Có thể chỉnh sửa thông tin sau khi trích xuất

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mô tả |
|-----------|-------|
| **Python 3.8+** | Ngôn ngữ lập trình chính |
| **OpenCV** | Xử lý ảnh, phát hiện và tiền xử lý |
| **Tesseract OCR** | Nhận dạng ký tự quang học (hỗ trợ tiếng Việt) |
| **face_recognition** | Nhận diện và mã hóa khuôn mặt (dlib-based) |
| **MySQL** | Cơ sở dữ liệu quan hệ |
| **Tkinter** | Giao diện desktop |
| **NumPy** | Xử lý mảng và tính toán |
| **Pillow (PIL)** | Xử lý ảnh Python |

---

## 📋 Yêu cầu hệ thống

### Phần mềm bắt buộc
- **Python**: 3.8 trở lên
- **MySQL Server**: 8.0 trở lên
- **Tesseract OCR**: Phiên bản mới nhất
- **CMake**: Cần thiết cho face-recognition/dlib

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

### Bước 2: Cài đặt các công cụ cần thiết

#### Trên macOS:
```bash
# Cài đặt CMake (bắt buộc cho face-recognition)
brew install cmake

# Cài đặt Tesseract OCR
brew install tesseract
brew install tesseract-lang  # Hỗ trợ tiếng Việt
```

#### Trên Ubuntu/Debian:
```bash
# Cài đặt CMake
sudo apt-get update
sudo apt-get install cmake

# Cài đặt Tesseract OCR
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-vie  # Hỗ trợ tiếng Việt
```

#### Trên Windows:
1. Tải và cài đặt CMake: https://cmake.org/download/
2. Tải và cài đặt Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
3. Thêm Tesseract vào PATH hệ thống

### Bước 3: Cài đặt Python dependencies

```bash
# Cài đặt các package cần thiết
pip install -r requirements.txt
```

**Lưu ý**: Nếu gặp lỗi thiếu setuptools hoặc wheel, chạy:
```bash
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### Bước 4: Cài đặt và cấu hình MySQL

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
Chỉnh sửa file `config/database.py`:

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

Từ thư mục gốc:
```bash
python src/main.py
```

Hoặc:
```bash
python main.py
```

### Hướng dẫn sử dụng

#### 1️⃣ Trích xuất thông tin từ thẻ sinh viên

1. Click vào nút **"Trích xuất thông tin"** ở cửa sổ chính
2. Click **"Chọn ảnh thẻ"** và chọn file ảnh thẻ sinh viên
3. Click **"Trích xuất thông tin"** để bắt đầu quá trình xử lý
4. Xem trước thông tin đã trích xuất:
   - MSSV, Họ tên, Ngày sinh
   - Niên khóa, Ngày hết hạn
   - Ảnh chân dung
5. Kiểm tra và chỉnh sửa thông tin nếu cần
6. Click **"Lưu vào database"** để lưu thông tin

> 💡 **Mẹo**: Ảnh càng rõ nét, độ chính xác OCR càng cao!

#### 2️⃣ Tìm kiếm sinh viên theo ảnh khuôn mặt

1. Click vào nút **"Tìm kiếm theo ảnh khuôn mặt"**
2. Click **"Chọn ảnh khuôn mặt"** và chọn ảnh cần tìm
3. Click **"Tìm kiếm"** để bắt đầu
4. Xem danh sách kết quả tìm được (sắp xếp theo độ tương đồng)
5. Click vào một sinh viên trong danh sách để xem chi tiết

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
│   │   ├── ocr_extractor.py    # Trích xuất text bằng OCR
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
| `nien_khoa` | VARCHAR(20) | Niên khóa |
| `ngay_het_han` | DATE | Thẻ có giá trị đến ngày |
| `avatar_path` | TEXT | Đường dẫn file ảnh chân dung |
| `face_encoding` | BLOB | Vector mã hóa khuôn mặt |
| `created_at` | TIMESTAMP | Thời gian tạo record |

### Indexes
- `idx_mssv`: Index trên cột `mssv` (tìm kiếm nhanh)
- `idx_ho_ten`: Index trên cột `ho_ten` (tìm kiếm theo tên)

---

## ⚙️ Các tính năng kỹ thuật

### 🔬 OCR Engine
- **Đa phương thức preprocessing**:
  - Grayscale cơ bản
  - Enhanced grayscale
  - OTSU binary threshold
  - Adaptive threshold
  - Inverted binary
  
- **Multiple PSM modes**: Thử nhiều chế độ PSM (Page Segmentation Mode) để tối ưu
- **Smart text scoring**: Tự động chọn kết quả OCR tốt nhất
- **Flexible regex parsing**: Xử lý OCR noise và typo linh hoạt

### 🤖 Face Recognition
- **Dual model support**: HOG (nhanh) và CNN (chính xác)
- **Auto resizing**: Tự động resize ảnh nhỏ để tăng độ chính xác
- **128-dimensional encoding**: Mã hóa khuôn mặt thành vector 128D
- **Distance-based matching**: So khớp dựa trên khoảng cách Euclidean

### 🖼️ Image Processing
- **Automatic card detection**: Tự động phát hiện vùng thẻ
- **Noise reduction**: Giảm nhiễu ảnh
- **Contrast enhancement**: Tăng cường độ tương phản
- **Smart cropping**: Cắt chính xác vùng quan tâm

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

### ❌ Lỗi: `tesseract is not installed or it's not in your PATH`
**Nguyên nhân**: Tesseract OCR chưa được cài đặt hoặc không có trong PATH

**Giải pháp**:
```bash
# macOS
brew install tesseract
brew install tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-vie

# Windows: Tải và cài từ https://github.com/UB-Mannheim/tesseract/wiki
# Sau đó thêm vào PATH hoặc chỉnh sửa trong code
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

2. Kiểm tra file `config/database.py` có đúng thông tin không
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

---

### ❌ Không phát hiện được khuôn mặt
**Nguyên nhân**: Ảnh quá nhỏ hoặc khuôn mặt không rõ

**Giải pháp**:
- Sử dụng ảnh có chất lượng tốt
- Đảm bảo khuôn mặt rõ ràng, không bị che khuất
- Code đã tự động resize ảnh nhỏ, nhưng vẫn nên dùng ảnh chất lượng tốt

---

## 📝 Lưu ý quan trọng

- ✅ **Ảnh chất lượng**: Ảnh thẻ càng rõ nét, độ chính xác OCR càng cao
- ✅ **Tiếng Việt**: Cần cài đặt Tesseract với hỗ trợ tiếng Việt (`tesseract-lang`)
- ✅ **Hiệu năng**: Face recognition sử dụng CNN model có thể chậm hơn, nhưng chính xác hơn
- ✅ **MySQL**: Đảm bảo MySQL đang chạy trước khi sử dụng ứng dụng
- ✅ **Backup**: Nên backup database thường xuyên

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
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Công cụ OCR
- [face_recognition](https://github.com/ageitgey/face_recognition) - Thư viện nhận diện khuôn mặt
- [dlib](http://dlib.net/) - Machine learning library

---

**⭐ Nếu dự án hữu ích, hãy cho một [star](https://github.com/nvn0205/student-card-extractor) nhé! ⭐**
