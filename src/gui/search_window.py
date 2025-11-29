"""Search window for face-based student search with real-time camera"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import threading
from PIL import Image, ImageTk
import numpy as np
from ..face_matching.face_matcher import search_by_face_image, get_similarity_score, encode_face
import face_recognition


class SearchWindow:
    """Window for searching students by face image with real-time camera"""
    
    # Modern color palette
    COLORS = {
        'primary': '#6366f1',
        'success': '#10b981',
        'danger': '#ef4444',
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
        self.root.title("🔍 Tìm kiếm sinh viên theo ảnh khuôn mặt - Camera Realtime")
        self.root.geometry("1100x780")
        self.root.configure(bg=self.COLORS['bg'])
        
        self.search_image = None
        self.search_results = []
        
        # Camera-related variables
        self.camera = None
        self.camera_active = False
        self.camera_thread = None
        self.current_frame = None
        self.last_face_detection_time = 0
        self.last_preview_update_time = 0
        self.face_detection_interval = 1.0  # Detect face every 1 second
        self.preview_update_interval = 0.1  # Update preview every 100ms (10 FPS for display)
        self.auto_search_enabled = True
        self.camera_lock = threading.Lock()  # Lock for camera operations
        self.updating_preview = False  # Flag to prevent concurrent updates
        
        self.create_widgets()
        
        # Cleanup when window closes
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
            text="🔍 Tìm kiếm Sinh viên theo Khuôn mặt",
            font=("Segoe UI", 16, "bold"),
            bg=self.COLORS['primary'],
            fg=self.COLORS['white']
        )
        header_title.pack(pady=20)
        
        # Toolbar
        toolbar_frame = tk.Frame(self.root, bg=self.COLORS['bg'], padx=15, pady=12)
        toolbar_frame.pack(fill=tk.X)
        
        controls_frame = tk.Frame(toolbar_frame, bg=self.COLORS['bg'])
        controls_frame.pack(side=tk.LEFT)
        
        self.camera_btn = tk.Button(
            controls_frame,
            text="📹 Bật Camera",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['success'],
            fg=self.COLORS['white'],
            command=self.toggle_camera,
            relief=tk.FLAT,
            cursor="hand2",
            padx=25,
            pady=10,
            activebackground=self.COLORS['success'],
            activeforeground=self.COLORS['white']
        )
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        upload_btn = tk.Button(
            controls_frame,
            text="📁 Chọn ảnh",
            font=("Segoe UI", 11),
            bg=self.COLORS['primary'],
            fg=self.COLORS['white'],
            command=self.upload_image,
            relief=tk.FLAT,
            cursor="hand2",
            padx=20,
            pady=10,
            activebackground=self.COLORS['primary'],
            activeforeground=self.COLORS['white']
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Auto-search toggle
        toggle_frame = tk.Frame(toolbar_frame, bg=self.COLORS['bg'])
        toggle_frame.pack(side=tk.LEFT, padx=20)
        
        self.auto_search_var = tk.BooleanVar(value=True)
        auto_search_check = tk.Checkbutton(
            toggle_frame,
            text="🤖 Tự động tìm kiếm khi phát hiện khuôn mặt",
            variable=self.auto_search_var,
            font=("Segoe UI", 10),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text'],
            selectcolor=self.COLORS['white'],
            activebackground=self.COLORS['bg'],
            activeforeground=self.COLORS['text'],
            command=self.toggle_auto_search,
            cursor="hand2"
        )
        auto_search_check.pack()
        
        # Status label
        status_frame = tk.Frame(toolbar_frame, bg=self.COLORS['bg'])
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        status_indicator = tk.Label(
            status_frame,
            text="●",
            font=("Arial", 12),
            bg=self.COLORS['bg'],
            fg=self.COLORS['success']
        )
        status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        
        self.status_label = tk.Label(
            status_frame,
            text="Sẵn sàng",
            font=("Segoe UI", 10),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Left frame - camera preview card
        left_card = tk.Frame(content_frame, bg=self.COLORS['card_bg'], width=480)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_card.pack_propagate(False)
        
        # Card border
        left_border = tk.Frame(left_card, bg='#e5e7eb', padx=2, pady=2)
        left_border.pack(fill=tk.BOTH, expand=True)
        
        left_content = tk.Frame(left_border, bg=self.COLORS['card_bg'])
        left_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        preview_title = tk.Label(
            left_content,
            text="📹 Camera Preview",
            font=("Segoe UI", 12, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['text']
        )
        preview_title.pack(pady=(0, 10))
        
        self.preview_label = tk.Label(
            left_content,
            text="Nhấn 'Bật Camera' để bắt đầu",
            font=("Segoe UI", 10),
            bg="#f9fafb",
            fg=self.COLORS['text_light'],
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Right frame - results card
        right_card = tk.Frame(content_frame, bg=self.COLORS['card_bg'])
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Card border
        right_border = tk.Frame(right_card, bg='#e5e7eb', padx=2, pady=2)
        right_border.pack(fill=tk.BOTH, expand=True)
        
        right_content = tk.Frame(right_border, bg=self.COLORS['card_bg'])
        right_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        results_title = tk.Label(
            right_content,
            text="📊 Kết quả tìm kiếm",
            font=("Segoe UI", 13, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['primary']
        )
        results_title.pack(fill=tk.X, pady=(0, 15))
        
        # Results listbox with scrollbar
        list_container = tk.Frame(right_content, bg=self.COLORS['white'], relief=tk.SUNKEN, borderwidth=1)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Modern Treeview
        self.results_tree = ttk.Treeview(
            list_container,
            columns=("MSSV", "Họ tên", "Ngày sinh", "Niên khóa", "Độ khớp"),
            show="headings",
            yscrollcommand=scrollbar.set,
            selectmode=tk.BROWSE,
            style="Modern.Treeview"
        )
        
        scrollbar.config(command=self.results_tree.yview)
        
        # Configure modern style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Modern.Treeview",
            background=self.COLORS['white'],
            foreground=self.COLORS['text'],
            fieldbackground=self.COLORS['white'],
            rowheight=35,
            font=("Segoe UI", 10)
        )
        style.configure(
            "Modern.Treeview.Heading",
            background=self.COLORS['light'],
            foreground=self.COLORS['text'],
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT
        )
        
        # Configure columns
        self.results_tree.heading("#0", text="ID")
        self.results_tree.heading("MSSV", text="Mã SV")
        self.results_tree.heading("Họ tên", text="Họ và Tên")
        self.results_tree.heading("Ngày sinh", text="Ngày Sinh")
        self.results_tree.heading("Niên khóa", text="Niên Khóa")
        self.results_tree.heading("Độ khớp", text="Độ Khớp (%)")
        
        self.results_tree.column("#0", width=50, anchor=tk.CENTER)
        self.results_tree.column("MSSV", width=110, anchor=tk.CENTER)
        self.results_tree.column("Họ tên", width=180, anchor=tk.W)
        self.results_tree.column("Ngày sinh", width=120, anchor=tk.CENTER)
        self.results_tree.column("Niên khóa", width=120, anchor=tk.CENTER)
        self.results_tree.column("Độ khớp", width=110, anchor=tk.CENTER)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.results_tree.bind("<<TreeviewSelect>>", self.on_result_select)
        
        # Detail card
        detail_card = tk.Frame(right_content, bg=self.COLORS['card_bg'], height=200)
        detail_card.pack(fill=tk.X, pady=(15, 0))
        detail_card.pack_propagate(False)
        
        detail_border = tk.Frame(detail_card, bg='#e5e7eb', padx=2, pady=2)
        detail_border.pack(fill=tk.BOTH, expand=True)
        
        detail_content = tk.Frame(detail_border, bg=self.COLORS['card_bg'])
        detail_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        detail_title = tk.Label(
            detail_content,
            text="ℹ️ Chi tiết",
            font=("Segoe UI", 12, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['text']
        )
        detail_title.pack(fill=tk.X, pady=(0, 10))
        
        detail_inner = tk.Frame(detail_content, bg=self.COLORS['white'], relief=tk.SUNKEN, borderwidth=1)
        detail_inner.pack(fill=tk.BOTH, expand=True)
        
        self.detail_text = tk.Text(
            detail_inner,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=self.COLORS['white'],
            fg=self.COLORS['text'],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        
        # Avatar display in detail
        self.avatar_label = tk.Label(
            detail_content,
            text="",
            bg=self.COLORS['card_bg']
        )
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture"""
        try:
            # Check if camera is already open
            if self.camera is not None:
                self.stop_camera()
            
            # Open camera
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera. Kiểm tra xem camera đã được kết nối chưa.")
                self.camera = None
                return
            
            # Set camera properties for better performance
            try:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
            except Exception as e:
                print(f"Warning: Could not set camera properties: {e}")
            
            # Reset timing variables
            import time
            self.last_face_detection_time = time.time()
            self.last_preview_update_time = time.time()
            
            with self.camera_lock:
                self.camera_active = True
            
            self.camera_btn.config(text="⏹️ Tắt Camera", bg=self.COLORS['danger'])
            self.update_status("Camera đang bật...")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", f"Lỗi khi khởi động camera: {str(e)}")
            with self.camera_lock:
                self.camera_active = False
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
    
    def stop_camera(self):
        """Stop camera capture"""
        with self.camera_lock:
            self.camera_active = False
        
        # Wait a bit for thread to stop
        if self.camera_thread and self.camera_thread.is_alive():
            import time
            time.sleep(0.2)
        
        # Release camera
        try:
            if self.camera:
                self.camera.release()
                self.camera = None
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        # Update UI
        self.camera_btn.config(text="📹 Bật Camera", bg=self.COLORS['success'])
        self.update_status("Camera đã tắt")
        
        # Clear preview
        try:
            self.preview_label.config(image="", text="Camera đã tắt")
            self.preview_label.image = None
            self.current_frame = None
        except Exception as e:
            print(f"Error clearing preview: {e}")
    
    def handle_camera_error(self, error_msg):
        """Handle camera errors in main thread"""
        self.stop_camera()
        messagebox.showerror("Lỗi Camera", f"Camera gặp lỗi: {error_msg}\nCamera đã được tắt.")
    
    def camera_loop(self):
        """Main camera loop running in separate thread"""
        import time
        
        try:
            while self.camera_active:
                try:
                    # Check if camera is still valid
                    if not self.camera or not self.camera.isOpened():
                        break
                    
                    ret, frame = self.camera.read()
                    
                    if not ret or frame is None:
                        time.sleep(0.1)
                        continue
                    
                    with self.camera_lock:
                        if not self.camera_active:
                            break
                        self.current_frame = frame.copy()
                    
                    # Update preview periodically (throttled to avoid overload)
                    current_time = time.time()
                    if current_time - self.last_preview_update_time >= self.preview_update_interval:
                        self.last_preview_update_time = current_time
                        
                        # Simple frame for preview (no face detection to avoid overhead)
                        display_frame = self.prepare_preview_frame(frame)
                        
                        # Update preview in main thread (only if not already updating)
                        if not self.updating_preview:
                            self.root.after(0, self.update_preview, display_frame)
                    
                    # Auto-search if enabled
                    if self.auto_search_var.get():
                        if current_time - self.last_face_detection_time >= self.face_detection_interval:
                            self.last_face_detection_time = current_time
                            # Search in separate thread to avoid blocking
                            threading.Thread(
                                target=self.auto_search_face, 
                                args=(frame.copy(),), 
                                daemon=True
                            ).start()
                    
                    # Small delay to control frame rate
                    time.sleep(0.03)  # ~30 FPS for capture
                    
                except Exception as e:
                    print(f"Error in camera loop iteration: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)  # Wait before retrying
                    
        except Exception as e:
            print(f"Fatal error in camera loop: {e}")
            import traceback
            traceback.print_exc()
            # Notify main thread of error
            self.root.after(0, lambda: self.handle_camera_error(str(e)))
        finally:
            # Cleanup
            with self.camera_lock:
                self.camera_active = False
    
    def prepare_preview_frame(self, frame):
        """Prepare frame for preview display (simple, fast)"""
        try:
            return frame.copy()
        except Exception as e:
            print(f"Error preparing preview frame: {e}")
            return frame
    
    def process_frame_for_display(self, frame):
        """Process frame for display with face detection overlay"""
        try:
            display_frame = frame.copy()
            
            # Detect faces for visualization (lightweight)
            try:
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(
                    rgb_frame, 
                    model='hog', 
                    number_of_times_to_upsample=0
                )
                
                # Draw rectangles around detected faces
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face Detected", (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                # If face detection fails, just return the frame
                print(f"Face detection for display failed: {e}")
            
            return display_frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def update_preview(self, frame):
        """Update preview label with new frame (called from main thread)"""
        # Check if still active and not already updating
        if not self.camera_active or self.updating_preview:
            return
        
        try:
            self.updating_preview = True
            
            # Validate frame
            if frame is None or frame.size == 0:
                return
            
            # Resize for display
            height, width = frame.shape[:2]
            if height == 0 or width == 0:
                return
                
            max_width = 440
            max_height = 600
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                if new_width > 0 and new_height > 0:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    return
            
            # Convert to PIL
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                photo = ImageTk.PhotoImage(image=pil_image)
                
                # Update label only if camera is still active
                if self.camera_active:
                    self.preview_label.config(image=photo, text="")
                    self.preview_label.image = photo  # Keep reference
            except Exception as e:
                print(f"Error converting frame to PhotoImage: {e}")
            
        except Exception as e:
            print(f"Error updating preview: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.updating_preview = False
    
    def auto_search_face(self, frame):
        """Auto-search for faces in frame"""
        if not self.camera_active:
            return
        
        try:
            # Encode face
            face_encoding = encode_face(frame)
            
            if face_encoding is None:
                return
            
            # Search for matching students (tolerance 0.5 - chặt chẽ hơn)
            results = search_by_face_image(frame, tolerance=0.5, max_results=5)
            
            # Update results in main thread
            if results:
                self.root.after(0, self.update_search_results, results, frame.copy())
            else:
                # Clear results if no match found
                self.root.after(0, self.clear_results)
                
        except Exception as e:
            print(f"Error in auto search: {e}")
    
    def update_search_results(self, results, frame):
        """Update search results display"""
        if not self.camera_active:
            return
        
        self.search_results = results
        self.search_image = frame
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Display results - chỉ hiển thị những kết quả match hoặc có độ tương đồng cao
        for idx, result in enumerate(results):
            student = result['student']
            similarity = get_similarity_score(result['distance'])
            
            # Chỉ hiển thị nếu match hoặc similarity >= 60%
            if result['match'] or similarity >= 60:
                self.results_tree.insert(
                    "",
                    tk.END,
                    text=str(student['id']),
                    values=(
                        student.get('mssv', ''),
                        student.get('ho_ten', ''),
                        str(student.get('ngay_sinh', '')) if student.get('ngay_sinh') else '',
                        student.get('nien_khoa', ''),
                        f"{similarity:.1f}%"
                    ),
                    tags=('matched' if result['match'] else 'similar',)
                )
        
        # Configure tags with modern colors
        self.results_tree.tag_configure('matched', background='#d1fae5')  # Light green
        self.results_tree.tag_configure('similar', background='#fef3c7')  # Light amber
        
        # Select first result if available
        children = self.results_tree.get_children()
        if children:
            first_item = children[0]
            self.results_tree.selection_set(first_item)
            self.results_tree.focus(first_item)
            self.on_result_select(None)
        
        # Update status
        if results:
            best_match = results[0]
            similarity = get_similarity_score(best_match['distance'])
            self.update_status(f"Tìm thấy {len(results)} kết quả (Độ khớp tốt nhất: {similarity}%)")
    
    def clear_results(self):
        """Clear search results"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.config(state=tk.DISABLED)
        self.avatar_label.place_forget()
    
    def toggle_auto_search(self):
        """Toggle auto-search mode"""
        self.auto_search_enabled = self.auto_search_var.get()
        if self.auto_search_enabled:
            self.update_status("Tự động tìm kiếm đã bật")
        else:
            self.update_status("Tự động tìm kiếm đã tắt")
    
    def upload_image(self):
        """Upload search image (backup method)"""
        # Stop camera if running
        if self.camera_active:
            self.stop_camera()
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh khuôn mặt",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_search_image(file_path)
            if self.status_callback:
                self.status_callback(f"Đã tải ảnh: {os.path.basename(file_path)}")
    
    def load_search_image(self, path):
        """Load and display search image"""
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh")
            return
        
        self.search_image = image.copy()
        
        # Search immediately
        threading.Thread(target=self.search_students_async, daemon=True).start()
    
    def search_students_async(self):
        """Search students asynchronously"""
        if self.search_image is None:
            return
        
        self.root.after(0, self.update_status, "Đang tìm kiếm...")
        
        try:
            results = search_by_face_image(self.search_image, tolerance=0.5, max_results=5)
            self.search_results = results
            
            self.root.after(0, self.display_uploaded_results, results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi khi tìm kiếm: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Lỗi khi tìm kiếm"))
    
    def display_uploaded_results(self, results):
        """Display results from uploaded image"""
        # Update preview
        display_image = self.resize_for_display(self.search_image, max_width=440, max_height=600)
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if not results:
            messagebox.showinfo("Kết quả", "Không tìm thấy sinh viên khớp")
            self.update_status("Không tìm thấy kết quả")
            return
        
        # Add results to treeview - chỉ hiển thị những kết quả match hoặc có độ tương đồng cao
        for idx, result in enumerate(results):
            student = result['student']
            similarity = get_similarity_score(result['distance'])
            
            # Chỉ hiển thị nếu match hoặc similarity >= 60%
            if result['match'] or similarity >= 60:
                self.results_tree.insert(
                    "",
                    tk.END,
                    text=str(student['id']),
                    values=(
                        student.get('mssv', ''),
                        student.get('ho_ten', ''),
                        str(student.get('ngay_sinh', '')) if student.get('ngay_sinh') else '',
                        student.get('nien_khoa', ''),
                        f"{similarity:.1f}%"
                    ),
                    tags=('matched' if result['match'] else 'similar',)
                )
        
        # Configure tags with modern colors
        self.results_tree.tag_configure('matched', background='#d1fae5')  # Light green
        self.results_tree.tag_configure('similar', background='#fef3c7')  # Light amber
        
        # Select first result
        first_item = self.results_tree.get_children()[0]
        self.results_tree.selection_set(first_item)
        self.results_tree.focus(first_item)
        self.on_result_select(None)
        
        self.update_status(f"Tìm thấy {len(results)} kết quả")
    
    def resize_for_display(self, image, max_width=440, max_height=600):
        """Resize image for display"""
        height, width = image.shape[:2]
        
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def on_result_select(self, event):
        """Handle result selection"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_id = int(self.results_tree.item(item, "text"))
        
        # Find student in results
        selected_student = None
        for result in self.search_results:
            if result['student']['id'] == item_id:
                selected_student = result['student']
                break
        
        if selected_student:
            self.display_student_detail(selected_student)
    
    def display_student_detail(self, student):
        """Display student detail"""
        # Update detail text
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        
        detail_info = f"""MSSV: {student.get('mssv', 'N/A')}
Họ tên: {student.get('ho_ten', 'N/A')}
Ngày sinh: {student.get('ngay_sinh', 'N/A')}
Niên khóa: {student.get('nien_khoa', 'N/A')}
Thẻ có giá trị đến: {student.get('ngay_het_han', 'N/A')}
"""
        
        self.detail_text.insert(1.0, detail_info)
        self.detail_text.config(state=tk.DISABLED)
        
        # Display avatar if available
        avatar_path = student.get('avatar_path')
        if avatar_path and os.path.exists(avatar_path):
            avatar_image = cv2.imread(avatar_path)
            if avatar_image is not None:
                # Resize avatar
                avatar_resized = self.resize_for_display(avatar_image, max_width=150, max_height=180)
                
                # Convert to PIL
                avatar_rgb = cv2.cvtColor(avatar_resized, cv2.COLOR_BGR2RGB)
                pil_avatar = Image.fromarray(avatar_rgb)
                photo = ImageTk.PhotoImage(image=pil_avatar)
                
                # Update avatar label
                self.avatar_label.config(image=photo, text="")
                self.avatar_label.image = photo
                
                # Place avatar in detail frame
                self.avatar_label.place(relx=0.7, rely=0.15, anchor=tk.NW)
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        if self.status_callback:
            self.status_callback(message)
    
    def on_closing(self):
        """Handle window closing"""
        try:
            if self.camera_active:
                self.stop_camera()
        except Exception as e:
            print(f"Error stopping camera on close: {e}")
        
        # Give time for cleanup
        import time
        time.sleep(0.1)
        
        try:
            self.root.destroy()
        except Exception as e:
            print(f"Error destroying window: {e}")
