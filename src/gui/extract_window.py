"""Extract window for processing student cards"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import threading
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
    
    # Modern color palette
    COLORS = {
        'primary': '#6366f1',
        'success': '#10b981',
        'warning': '#f59e0b',
        'dark': '#1f2937',
        'light': '#f9fafb',
        'white': '#ffffff',
        'bg': '#f3f4f6',
        'card_bg': '#ffffff',
        'text': '#111827',
        'text_light': '#6b7280',
    }
    
    def __init__(self, root, status_callback=None):
        self.root = root
        self.status_callback = status_callback
        self.root.title("📄 Trích xuất thông tin từ thẻ sinh viên")
        self.root.geometry("1100x750")
        self.root.configure(bg=self.COLORS['bg'])
        
        self.image_path = None
        self.card_image = None
        self.extracted_info = {}
        self.loading_window = None
        
        # Create avatars directory
        self.avatars_dir = "avatars"
        os.makedirs(self.avatars_dir, exist_ok=True)
        
        self.create_widgets()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create modern UI widgets"""
        # Header
        header_frame = tk.Frame(
            self.root,
            bg=self.COLORS['primary'],
            height=80
        )
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_content = tk.Frame(header_frame, bg=self.COLORS['primary'])
        header_content.pack(expand=True)
        
        header_title = tk.Label(
            header_content,
            text="📄 Trích xuất Thông tin Thẻ Sinh viên",
            font=("Segoe UI", 16, "bold"),
            bg=self.COLORS['primary'],
            fg=self.COLORS['white']
        )
        header_title.pack(pady=20)
        
        # Top toolbar - buttons
        toolbar_frame = tk.Frame(self.root, bg=self.COLORS['bg'], padx=15, pady=12)
        toolbar_frame.pack(fill=tk.X)
        
        button_container = tk.Frame(toolbar_frame, bg=self.COLORS['bg'])
        button_container.pack()
        
        upload_btn = tk.Button(
            button_container,
            text="📁 Chọn ảnh thẻ",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['primary'],
            fg=self.COLORS['white'],
            command=self.upload_image,
            relief=tk.FLAT,
            cursor="hand2",
            padx=25,
            pady=10,
            activebackground=self.COLORS['primary'],
            activeforeground=self.COLORS['white']
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        extract_btn = tk.Button(
            button_container,
            text="⚡ Trích xuất thông tin",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['success'],
            fg=self.COLORS['white'],
            command=self.extract_info,
            relief=tk.FLAT,
            cursor="hand2",
            padx=25,
            pady=10,
            state=tk.DISABLED,
            activebackground=self.COLORS['success'],
            activeforeground=self.COLORS['white']
        )
        extract_btn.pack(side=tk.LEFT, padx=5)
        self.extract_btn = extract_btn
        
        save_btn = tk.Button(
            button_container,
            text="💾 Lưu vào database",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['warning'],
            fg=self.COLORS['white'],
            command=self.save_to_database,
            relief=tk.FLAT,
            cursor="hand2",
            padx=25,
            pady=10,
            state=tk.DISABLED,
            activebackground=self.COLORS['warning'],
            activeforeground=self.COLORS['white']
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn = save_btn
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Left frame - image preview card
        left_card = tk.Frame(content_frame, bg=self.COLORS['card_bg'], relief=tk.FLAT)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Card border effect
        left_border = tk.Frame(left_card, bg='#e5e7eb', padx=2, pady=2)
        left_border.pack(fill=tk.BOTH, expand=True)
        
        left_content = tk.Frame(left_border, bg=self.COLORS['card_bg'])
        left_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        image_title = tk.Label(
            left_content,
            text="📷 Ảnh thẻ sinh viên",
            font=("Segoe UI", 12, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['text']
        )
        image_title.pack(pady=(0, 10))
        
        image_label = tk.Label(
            left_content,
            text="📷 Chưa có ảnh\n\nKích thước: Chưa xác định",
            font=("Segoe UI", 10),
            bg="#f9fafb",
            fg=self.COLORS['text_light'],
            relief=tk.SUNKEN,
            borderwidth=2
        )
        image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label = image_label
        
        # Right frame - extracted info card
        right_card = tk.Frame(content_frame, bg=self.COLORS['card_bg'], width=450)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_card.pack_propagate(False)
        
        # Card border
        right_border = tk.Frame(right_card, bg='#e5e7eb', padx=2, pady=2)
        right_border.pack(fill=tk.BOTH, expand=True)
        
        right_content = tk.Frame(right_border, bg=self.COLORS['card_bg'])
        right_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        info_title = tk.Label(
            right_content,
            text="📝 Thông tin trích xuất",
            font=("Segoe UI", 13, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['primary']
        )
        info_title.pack(fill=tk.X, pady=(0, 15))
        
        # Info fields
        info_frame = tk.Frame(right_content, bg=self.COLORS['card_bg'])
        info_frame.pack(fill=tk.X)
        
        fields = [
            ("MSSV", "mssv", "🔢"),
            ("Họ tên", "ho_ten", "👤"),
            ("Ngày sinh", "ngay_sinh", "📅"),
            ("Niên khóa", "nien_khoa", "🎓"),
            ("Thẻ có giá trị đến", "ngay_het_han", "⏰")
        ]
        
        self.info_entries = {}
        
        for label_text, field_name, icon in fields:
            field_container = tk.Frame(info_frame, bg=self.COLORS['card_bg'], pady=8)
            field_container.pack(fill=tk.X, pady=3)
            
            label_frame = tk.Frame(field_container, bg=self.COLORS['card_bg'])
            label_frame.pack(fill=tk.X, pady=(0, 5))
            
            label = tk.Label(
                label_frame,
                text=f"{icon} {label_text}:",
                font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg'],
                fg=self.COLORS['text'],
                anchor=tk.W
            )
            label.pack(side=tk.LEFT)
            
            entry = tk.Entry(
                field_container,
                font=("Segoe UI", 10),
                relief=tk.FLAT,
                borderwidth=1,
                highlightthickness=1,
                highlightbackground="#d1d5db",
                highlightcolor=self.COLORS['primary'],
                bg=self.COLORS['white'],
                fg=self.COLORS['text']
            )
            entry.pack(fill=tk.X, ipady=6)
            self.info_entries[field_name] = entry
        
        # Avatar preview - larger display (moved before OCR text)
        avatar_section = tk.Frame(right_content, bg=self.COLORS['card_bg'], pady=15)
        avatar_section.pack(fill=tk.X, pady=(0, 10))
        
        avatar_title = tk.Label(
            avatar_section,
            text="📸 Ảnh chân dung",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['text']
        )
        avatar_title.pack(anchor=tk.W, pady=(0, 8))
        
        avatar_image_frame = tk.Frame(
            avatar_section,
            bg="#f9fafb",
            relief=tk.SUNKEN,
            borderwidth=2,
            height=350  # Tăng chiều cao để hiển thị đủ toàn bộ ảnh
        )
        avatar_image_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        avatar_image_frame.pack_propagate(False)
        
        # Container to center the image
        avatar_container = tk.Frame(avatar_image_frame, bg="#f9fafb")
        avatar_container.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
        
        self.avatar_label = tk.Label(
            avatar_container,
            text="Chưa có ảnh",
            bg="#f9fafb",
            fg=self.COLORS['text_light'],
            font=("Segoe UI", 10),
            anchor=tk.CENTER
        )
        self.avatar_label.pack(expand=True)
        
        # Raw OCR text (collapsible/compact)
        raw_text_frame = tk.Frame(right_content, bg=self.COLORS['card_bg'])
        raw_text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        raw_text_header = tk.Frame(raw_text_frame, bg=self.COLORS['card_bg'])
        raw_text_header.pack(fill=tk.X)
        
        raw_text_label = tk.Label(
            raw_text_header,
            text="🔍 Text từ OCR (để kiểm tra)",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['text_light']
        )
        raw_text_label.pack(anchor=tk.W)
        
        raw_text_container = tk.Frame(raw_text_frame, bg=self.COLORS['white'], relief=tk.SUNKEN, borderwidth=1)
        raw_text_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        raw_text_scrollbar = tk.Scrollbar(raw_text_container)
        raw_text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.raw_text_text = tk.Text(
            raw_text_container,
            font=("Consolas", 8),
            wrap=tk.WORD,
            yscrollcommand=raw_text_scrollbar.set,
            bg=self.COLORS['white'],
            fg=self.COLORS['text'],
            relief=tk.FLAT,
            state=tk.DISABLED,
            padx=8,
            pady=8
        )
        self.raw_text_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        raw_text_scrollbar.config(command=self.raw_text_text.yview)
    
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
    
    def show_loading_dialog(self):
        """Show loading dialog"""
        if self.loading_window is not None:
            return
        
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Đang xử lý...")
        self.loading_window.geometry("350x180")
        self.loading_window.configure(bg=self.COLORS['bg'])
        self.loading_window.resizable(False, False)
        self.loading_window.transient(self.root)  # Make it modal
        self.loading_window.grab_set()  # Make it modal
        
        # Center on parent window
        self.loading_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (350 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (180 // 2)
        self.loading_window.geometry(f"350x180+{x}+{y}")
        
        # Content frame
        content = tk.Frame(self.loading_window, bg=self.COLORS['bg'], padx=30, pady=30)
        content.pack(expand=True, fill=tk.BOTH)
        
        # Spinner/loading icon
        spinner_label = tk.Label(
            content,
            text="⏳",
            font=("Segoe UI Emoji", 32),
            bg=self.COLORS['bg']
        )
        spinner_label.pack(pady=(0, 15))
        
        # Message
        message_label = tk.Label(
            content,
            text="Đang trích xuất thông tin...",
            font=("Segoe UI", 12, "bold"),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        message_label.pack(pady=(0, 10))
        
        # Sub message
        sub_label = tk.Label(
            content,
            text="Vui lòng đợi trong giây lát",
            font=("Segoe UI", 10),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text_light']
        )
        sub_label.pack()
        
        # Animate spinner
        self.animate_spinner(spinner_label)
        
        # Force update
        self.loading_window.update()
    
    def animate_spinner(self, label):
        """Animate spinner icon"""
        if self.loading_window is None or not self.loading_window.winfo_exists():
            return
        
        spinners = ["⏳", "⏰", "⏳", "⏰"]
        current = getattr(self, '_spinner_index', 0)
        label.config(text=spinners[current % len(spinners)])
        self._spinner_index = (current + 1) % len(spinners)
        
        # Schedule next animation
        if self.loading_window and self.loading_window.winfo_exists():
            self.loading_window.after(500, lambda: self.animate_spinner(label))
    
    def close_loading_dialog(self):
        """Close loading dialog"""
        if self.loading_window is not None:
            try:
                self.loading_window.destroy()
            except:
                pass
            self.loading_window = None
    
    def extract_info(self):
        """Extract information from card"""
        if self.card_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước")
            return
        
        # Show loading dialog
        self.show_loading_dialog()
        
        if self.status_callback:
            self.status_callback("Đang xử lý ảnh...")
        
        # Run extraction in separate thread to keep UI responsive
        threading.Thread(target=self._extract_info_thread, daemon=True).start()
    
    def _extract_info_thread(self):
        """Extract info in background thread"""
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
                padding=30  # Tăng padding để lấy đủ phần đầu và cổ
            )
            
            # Nếu không tìm thấy trên ảnh đã detect, thử trên ảnh gốc
            if face_image is None:
                print("⚠ No face found in detected card, trying original image...")
                face_image, face_location = extract_face_region(
                    self.card_image,
                    padding=30  # Tăng padding để lấy đủ phần đầu và cổ
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
            
            # Update UI in main thread
            self.root.after(0, self._update_ui_after_extraction, face_image)
            
        except Exception as e:
            # Close loading and show error in main thread
            self.root.after(0, self._handle_extraction_error, str(e))
    
    def _update_ui_after_extraction(self, face_image):
        """Update UI after extraction completes"""
        # Close loading dialog
        self.close_loading_dialog()
        
        try:
            # Display avatar if available
            if face_image is not None:
                self.display_avatar(face_image)
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
            
            # Update entry fields
            for field_name, entry in self.info_entries.items():
                value = self.extracted_info.get(field_name, '')
                entry.delete(0, tk.END)
                if value:
                    entry.insert(0, str(value))
                    print(f"Updated {field_name}: {value}")
                else:
                    print(f"No value for {field_name}")
            
            # Force UI refresh
            self.root.update_idletasks()
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
            messagebox.showerror("Lỗi", f"Lỗi khi cập nhật UI: {str(e)}")
    
    def _handle_extraction_error(self, error_msg):
        """Handle extraction error"""
        self.close_loading_dialog()
        messagebox.showerror("Lỗi", f"Lỗi khi trích xuất: {error_msg}")
        if self.status_callback:
            self.status_callback("Lỗi khi trích xuất")
    
    def display_avatar(self, face_image):
        """Display extracted face image - hiển thị toàn bộ ảnh không bị cắt"""
        try:
            print(f"📸 Displaying avatar, original size: {face_image.shape}")
            
            # Kiểm tra ảnh có hợp lệ không
            if face_image is None or face_image.size == 0:
                print("❌ Face image is empty or None")
                self.avatar_label.config(image="", text="Không thể hiển thị ảnh")
                return
            
            # Lấy kích thước ảnh gốc
            h, w = face_image.shape[:2]
            aspect_ratio = w / h if h > 0 else 1.0
            
            print(f"  Original aspect ratio: {aspect_ratio:.2f} (w/h)")
            
            # Kích thước frame có sẵn (frame height = 350, trừ padding 15*2 = 30, còn lại ~320px)
            # Frame width phụ thuộc vào right_card width (450px) trừ padding
            max_display_width = 380   # Chiều rộng tối đa có thể hiển thị
            max_display_height = 320  # Chiều cao tối đa có thể hiển thị (350 - 30 padding)
            
            # Tính scale để fit vào frame mà KHÔNG bị cắt, giữ nguyên tỷ lệ
            scale_w = max_display_width / w if w > 0 else 1
            scale_h = max_display_height / h if h > 0 else 1
            scale = min(scale_w, scale_h)  # Lấy scale nhỏ hơn để đảm bảo fit cả 2 chiều
            
            # Đảm bảo ảnh không quá nhỏ (minimum size)
            min_display_width = 200
            min_display_height = 240
            
            # Nếu ảnh quá nhỏ sau khi scale, scale lên
            if w * scale < min_display_width or h * scale < min_display_height:
                scale_min_w = min_display_width / w if w > 0 else 1
                scale_min_h = min_display_height / h if h > 0 else 1
                min_scale = max(scale_min_w, scale_min_h)
                
                # Chỉ scale up nếu không vượt quá max
                if min_scale > scale:
                    scale = min(min_scale, scale)  # Không vượt quá scale ban đầu
            
            # Tính kích thước mới (giữ tỷ lệ)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Đảm bảo không vượt quá kích thước tối đa (tránh bị cắt)
            if new_w > max_display_width:
                scale = max_display_width / w
                new_w = max_display_width
                new_h = int(h * scale)
            
            if new_h > max_display_height:
                scale = max_display_height / h
                new_h = max_display_height
                new_w = int(w * scale)
            
            print(f"  Scale factor: {scale:.2f}x")
            print(f"  Display size: {new_w}x{new_h}")
            
            # Resize ảnh với interpolation tốt
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
            
            print("✓ Avatar displayed successfully (full image, no cropping)")
            
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
        self.image_label.config(image="", text="📷 Chưa có ảnh\n\nKích thước: Chưa xác định")
        self.avatar_label.config(image="", text="Chưa có ảnh")
        self.extract_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing"""
        # Close loading dialog if open
        self.close_loading_dialog()
        self.root.destroy()

