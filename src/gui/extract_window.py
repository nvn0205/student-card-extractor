"""Extract window for processing student cards"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
from ..image_processing.card_detector import detect_and_extract_card
from ..extraction.ocr_extractor import extract_student_info
from ..extraction.face_extractor import extract_face_region, get_face_encoding_from_card
from ..face_matching.face_matcher import encode_face
from ..database.student_dao import StudentDAO


class ExtractWindow:
    """Window for extracting student information from card image"""
    
    def __init__(self, root, status_callback=None):
        self.root = root
        self.status_callback = status_callback
        self.root.title("Trích xuất thông tin từ thẻ sinh viên")
        self.root.geometry("1000x700")
        
        self.image_path = None
        self.card_image = None
        self.extracted_info = {}
        
        # Create avatars directory
        self.avatars_dir = "avatars"
        os.makedirs(self.avatars_dir, exist_ok=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create UI widgets"""
        # Top frame - buttons
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        upload_btn = tk.Button(
            top_frame,
            text="Chọn ảnh thẻ",
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            command=self.upload_image,
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        extract_btn = tk.Button(
            top_frame,
            text="Trích xuất thông tin",
            font=("Arial", 12),
            bg="#27ae60",
            fg="white",
            command=self.extract_info,
            relief=tk.FLAT,
            padx=20,
            pady=5,
            state=tk.DISABLED
        )
        extract_btn.pack(side=tk.LEFT, padx=5)
        self.extract_btn = extract_btn
        
        save_btn = tk.Button(
            top_frame,
            text="Lưu vào database",
            font=("Arial", 12),
            bg="#e67e22",
            fg="white",
            command=self.save_to_database,
            relief=tk.FLAT,
            padx=20,
            pady=5,
            state=tk.DISABLED
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn = save_btn
        
        # Main content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - image preview
        left_frame = tk.Frame(content_frame, width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        image_label = tk.Label(
            left_frame,
            text="Chưa có ảnh",
            font=("Arial", 12),
            bg="#ecf0f1",
            relief=tk.SUNKEN
        )
        image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label = image_label
        
        # Right frame - extracted info
        right_frame = tk.Frame(content_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        info_label = tk.Label(
            right_frame,
            text="Thông tin trích xuất",
            font=("Arial", 14, "bold"),
            bg="#34495e",
            fg="white"
        )
        info_label.pack(fill=tk.X, pady=(0, 10))
        
        # Info fields frame (không expand để có chỗ cho các phần khác)
        info_frame = tk.Frame(right_frame, bg="#ecf0f1")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create entry fields
        fields = [
            ("MSSV:", "mssv"),
            ("Họ tên:", "ho_ten"),
            ("Ngày sinh:", "ngay_sinh"),
            ("Niên khóa:", "nien_khoa"),
            ("Thẻ có giá trị đến:", "ngay_het_han")
        ]
        
        self.info_entries = {}
        
        for label_text, field_name in fields:
            field_frame = tk.Frame(info_frame, bg="#ecf0f1", pady=5)
            field_frame.pack(fill=tk.X, padx=5, pady=2)
            
            label = tk.Label(
                field_frame,
                text=label_text,
                font=("Arial", 11),
                bg="#ecf0f1",
                width=16,
                anchor=tk.W
            )
            label.pack(side=tk.LEFT)
            
            entry = tk.Entry(
                field_frame,
                font=("Arial", 11),
                width=25
            )
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.info_entries[field_name] = entry
        
        # Scrollable text area for raw OCR text (for debugging)
        raw_text_frame = tk.Frame(right_frame, bg="#ecf0f1")
        raw_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        raw_text_label = tk.Label(
            raw_text_frame,
            text="Text từ OCR (để kiểm tra):",
            font=("Arial", 10, "bold"),
            bg="#ecf0f1"
        )
        raw_text_label.pack(anchor=tk.W)
        
        raw_text_container = tk.Frame(raw_text_frame)
        raw_text_container.pack(fill=tk.BOTH, expand=True)
        
        raw_text_scrollbar = tk.Scrollbar(raw_text_container)
        raw_text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.raw_text_text = tk.Text(
            raw_text_container,
            font=("Arial", 9),
            wrap=tk.WORD,
            height=5,
            yscrollcommand=raw_text_scrollbar.set,
            bg="white",
            relief=tk.SUNKEN,
            state=tk.DISABLED
        )
        self.raw_text_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        raw_text_scrollbar.config(command=self.raw_text_text.yview)
        
        # Avatar preview - đặt ở cuối để dễ thấy hơn
        avatar_frame = tk.Frame(right_frame, bg="#ecf0f1", pady=10)
        avatar_frame.pack(fill=tk.X, padx=10, pady=5)
        
        avatar_title = tk.Label(
            avatar_frame,
            text="Ảnh chân dung",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1"
        )
        avatar_title.pack(anchor=tk.W)
        
        # Frame chứa ảnh avatar với border
        avatar_image_frame = tk.Frame(avatar_frame, bg="white", relief=tk.SUNKEN, borderwidth=2)
        avatar_image_frame.pack(pady=5, fill=tk.X)
        
        self.avatar_label = tk.Label(
            avatar_image_frame,
            text="Chưa có ảnh",
            bg="white",
            anchor=tk.CENTER,
            font=("Arial", 9)
        )
        # Không set width/height cố định, để ảnh tự điều chỉnh
        self.avatar_label.pack(padx=5, pady=5)
    
    def upload_image(self):
        """Upload and display image"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh thẻ sinh viên",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_image(file_path)
            self.extract_btn.config(state=tk.NORMAL)
            if self.status_callback:
                self.status_callback(f"Đã tải ảnh: {os.path.basename(file_path)}")
    
    def load_image(self, path):
        """Load and display image"""
        # Load image
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh")
            return
        
        self.card_image = image.copy()
        
        # Resize for display
        display_image = self.resize_for_display(image, max_width=480, max_height=600)
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update label
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def resize_for_display(self, image, max_width=480, max_height=600):
        """Resize image for display"""
        height, width = image.shape[:2]
        
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def extract_info(self):
        """Extract information from card"""
        if self.card_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước")
            return
        
        if self.status_callback:
            self.status_callback("Đang xử lý ảnh...")
        
        try:
            # Detect and extract card
            card_extracted, success = detect_and_extract_card(self.card_image)
            
            if not success:
                card_extracted = self.card_image  # Use original if detection fails
                print("⚠ Card detection failed, using original image")
            else:
                print("✓ Card detection successful")
            
            # Extract text info - thử với ảnh đã detect trước
            self.extracted_info = extract_student_info(card_extracted)
            
            # Nếu không trích xuất được thông tin, thử với ảnh gốc
            extracted_count = sum(1 for k in ['mssv', 'ho_ten', 'ngay_sinh', 'nien_khoa', 'ngay_het_han'] 
                                 if self.extracted_info.get(k))
            
            if extracted_count < 2 and success:  # Nếu ít hơn 2 field và detection đã thành công
                print("⚠ Low extraction rate, trying with original image...")
                original_info = extract_student_info(self.card_image)
                original_count = sum(1 for k in ['mssv', 'ho_ten', 'ngay_sinh', 'nien_khoa', 'ngay_het_han'] 
                                    if original_info.get(k))
                
                # Nếu ảnh gốc cho kết quả tốt hơn, dùng nó
                if original_count > extracted_count:
                    print(f"✓ Using original image (extracted {original_count} fields vs {extracted_count})")
                    self.extracted_info = original_info
                    card_extracted = self.card_image
            
            # Debug: Print extracted info
            print("=== Extracted Info ===")
            for key, value in self.extracted_info.items():
                if key != 'raw_text':
                    print(f"{key}: {value}")
            print("======================")
            
            # Extract face
            self.extracted_info['face_image'] = None
            self.extracted_info['face_encoding'] = None
            
            # Try to extract face from detected card first
            print("🔍 Trying to extract face from detected card...")
            face_image, face_location = extract_face_region(
                card_extracted,
                padding=20
            )
            
            # Nếu không tìm thấy trên ảnh đã detect, thử trên ảnh gốc
            if face_image is None:
                print("⚠ No face found in detected card, trying original image...")
                face_image, face_location = extract_face_region(
                    self.card_image,
                    padding=20
                )
            
            if face_image is not None:
                print(f"✓ Face extracted successfully! Size: {face_image.shape}")
                self.extracted_info['face_image'] = face_image
                # Get face encoding
                face_encoding = get_face_encoding_from_card(card_extracted)
                if face_encoding is None:
                    # Thử với ảnh gốc nếu không tìm thấy trên card detected
                    face_encoding = get_face_encoding_from_card(self.card_image)
                
                if face_encoding is not None:
                    self.extracted_info['face_encoding'] = face_encoding
                    print("✓ Face encoding generated")
                
                # Display avatar
                self.display_avatar(face_image)
                # Force UI update immediately after displaying avatar
                self.root.update_idletasks()
                self.root.update()
            else:
                print("❌ Could not extract face from image")
                self.avatar_label.config(image="", text="Không tìm thấy ảnh chân dung")
            
            # Display raw OCR text
            raw_text = self.extracted_info.get('raw_text', '')
            if raw_text:
                self.raw_text_text.config(state=tk.NORMAL)
                self.raw_text_text.delete(1.0, tk.END)
                self.raw_text_text.insert(1.0, raw_text)
                self.raw_text_text.config(state=tk.DISABLED)
            
            # Update entry fields - force update UI
            self.root.update_idletasks()
            for field_name, entry in self.info_entries.items():
                value = self.extracted_info.get(field_name, '')
                entry.delete(0, tk.END)
                if value:
                    entry.insert(0, str(value))
                    print(f"Updated {field_name}: {value}")
                else:
                    print(f"No value for {field_name}")
            
            # Force UI refresh
            self.root.update()
            
            self.save_btn.config(state=tk.NORMAL)
            
            if self.status_callback:
                self.status_callback("Trích xuất thông tin thành công")
            
            # Show info about what was extracted
            extracted_count = sum(1 for k in ['mssv', 'ho_ten', 'ngay_sinh', 'nien_khoa', 'ngay_het_han'] 
                                 if self.extracted_info.get(k))
            messagebox.showinfo(
                "Thành công", 
                f"Đã trích xuất {extracted_count}/5 trường thông tin từ thẻ.\n"
                "Vui lòng kiểm tra và chỉnh sửa nếu cần."
            )
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi trích xuất: {str(e)}")
            if self.status_callback:
                self.status_callback("Lỗi khi trích xuất")
    
    def display_avatar(self, face_image):
        """Display extracted face image"""
        try:
            print(f"📸 Displaying avatar, original size: {face_image.shape}")
            
            # Kiểm tra ảnh có hợp lệ không
            if face_image is None or face_image.size == 0:
                print("❌ Face image is empty or None")
                self.avatar_label.config(image="", text="Không thể hiển thị ảnh")
                return
            
            # Resize for display - đảm bảo ảnh có kích thước hợp lý
            # Resize về kích thước cố định để hiển thị đẹp
            target_width = 150
            target_height = 200
            
            h, w = face_image.shape[:2]
            # Tính scale để giữ tỷ lệ
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_face = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            print(f"  Resized face from {face_image.shape} to {display_face.shape}")
            
            # Convert to PIL
            if len(display_face.shape) == 3:
                face_rgb = cv2.cvtColor(display_face, cv2.COLOR_BGR2RGB)
            else:
                # Nếu là grayscale, convert sang RGB
                face_rgb = cv2.cvtColor(display_face, cv2.COLOR_GRAY2RGB)
            
            pil_face = Image.fromarray(face_rgb)
            photo = ImageTk.PhotoImage(image=pil_face)
            
            print(f"  Photo image created: {photo.width()}x{photo.height()}")
            
            # Update label - clear text first
            self.avatar_label.config(image="", text="")
            # Clear old reference trước
            if hasattr(self.avatar_label, 'image'):
                self.avatar_label.image = None
            
            # Update with new image - KHÔNG set text để tránh che ảnh
            self.avatar_label.config(image=photo, text="")
            self.avatar_label.image = photo  # Keep reference to prevent garbage collection
            
            # Force UI update ngay lập tức
            self.avatar_label.update_idletasks()
            self.root.update_idletasks()
            self.root.update()
            
            print("✓ Avatar displayed successfully")
            
        except Exception as e:
            print(f"❌ Error displaying avatar: {e}")
            import traceback
            traceback.print_exc()
            self.avatar_label.config(image="", text=f"Lỗi: {str(e)}")
    
    def save_to_database(self):
        """Save extracted information to database"""
        # Get values from entries
        student_data = {}
        for field_name, entry in self.info_entries.items():
            value = entry.get().strip()
            if value:
                student_data[field_name] = value
        
        # Validate required fields
        if not student_data.get('mssv'):
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập MSSV")
            return
        
        if not student_data.get('ho_ten'):
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập Họ tên")
            return
        
        # Check if MSSV already exists
        existing = StudentDAO.get_by_mssv(student_data['mssv'])
        if existing:
            if not messagebox.askyesno("Xác nhận", f"MSSV {student_data['mssv']} đã tồn tại. Bạn có muốn cập nhật?"):
                return
        
        try:
            # Save face image
            avatar_path = None
            if self.extracted_info.get('face_image') is not None:
                mssv = student_data['mssv']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                avatar_filename = f"{mssv}_{timestamp}.jpg"
                avatar_path = os.path.join(self.avatars_dir, avatar_filename)
                cv2.imwrite(avatar_path, self.extracted_info['face_image'])
                student_data['avatar_path'] = avatar_path
            
            # Add face encoding
            if self.extracted_info.get('face_encoding') is not None:
                student_data['face_encoding'] = self.extracted_info['face_encoding']
            
            # Save to database
            if existing:
                # Update
                StudentDAO.update(existing['id'], student_data)
                messagebox.showinfo("Thành công", "Đã cập nhật thông tin sinh viên")
            else:
                # Insert
                student_id = StudentDAO.create(student_data)
                if student_id:
                    messagebox.showinfo("Thành công", f"Đã lưu sinh viên với ID: {student_id}")
            
            if self.status_callback:
                self.status_callback("Đã lưu vào database")
            
            # Reset form
            self.reset_form()
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi lưu: {str(e)}")
            if self.status_callback:
                self.status_callback("Lỗi khi lưu")
    
    def reset_form(self):
        """Reset form after saving"""
        for entry in self.info_entries.values():
            entry.delete(0, tk.END)
        self.image_path = None
        self.card_image = None
        self.extracted_info = {}
        self.image_label.config(image="", text="Chưa có ảnh")
        self.avatar_label.config(image="", text="Chưa có ảnh")
        self.extract_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

