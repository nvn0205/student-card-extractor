"""Search window for face-based student search"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
from PIL import Image, ImageTk
from ..face_matching.face_matcher import search_by_face_image, get_similarity_score


class SearchWindow:
    """Window for searching students by face image"""
    
    def __init__(self, root, status_callback=None):
        self.root = root
        self.status_callback = status_callback
        self.root.title("Tìm kiếm sinh viên theo ảnh khuôn mặt")
        self.root.geometry("900x700")
        
        self.search_image = None
        self.search_results = []
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create UI widgets"""
        # Top frame - upload button
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        upload_btn = tk.Button(
            top_frame,
            text="Chọn ảnh khuôn mặt",
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            command=self.upload_image,
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        search_btn = tk.Button(
            top_frame,
            text="Tìm kiếm",
            font=("Arial", 12),
            bg="#27ae60",
            fg="white",
            command=self.search_students,
            relief=tk.FLAT,
            padx=20,
            pady=5,
            state=tk.DISABLED
        )
        search_btn.pack(side=tk.LEFT, padx=5)
        self.search_btn = search_btn
        
        # Main content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - image preview
        left_frame = tk.Frame(content_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        left_frame.pack_propagate(False)
        
        preview_label = tk.Label(
            left_frame,
            text="Chưa có ảnh",
            font=("Arial", 12),
            bg="#ecf0f1",
            relief=tk.SUNKEN
        )
        preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_label = preview_label
        
        # Right frame - results
        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        results_label = tk.Label(
            right_frame,
            text="Kết quả tìm kiếm",
            font=("Arial", 14, "bold"),
            bg="#34495e",
            fg="white"
        )
        results_label.pack(fill=tk.X, pady=(0, 10))
        
        # Results listbox with scrollbar
        list_frame = tk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview for results
        self.results_tree = ttk.Treeview(
            list_frame,
            columns=("MSSV", "Họ tên", "Ngày sinh", "Niên khóa", "Độ khớp"),
            show="headings",
            yscrollcommand=scrollbar.set,
            selectmode=tk.BROWSE
        )
        
        scrollbar.config(command=self.results_tree.yview)
        
        # Configure columns
        self.results_tree.heading("#0", text="ID")
        self.results_tree.heading("MSSV", text="MSSV")
        self.results_tree.heading("Họ tên", text="Họ tên")
        self.results_tree.heading("Ngày sinh", text="Ngày sinh")
        self.results_tree.heading("Niên khóa", text="Niên khóa")
        self.results_tree.heading("Độ khớp", text="Độ khớp (%)")
        
        self.results_tree.column("#0", width=40)
        self.results_tree.column("MSSV", width=100)
        self.results_tree.column("Họ tên", width=150)
        self.results_tree.column("Ngày sinh", width=100)
        self.results_tree.column("Niên khóa", width=120)
        self.results_tree.column("Độ khớp", width=100)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.results_tree.bind("<<TreeviewSelect>>", self.on_result_select)
        
        # Detail frame
        detail_frame = tk.Frame(right_frame, bg="#ecf0f1", height=200)
        detail_frame.pack(fill=tk.X, pady=(10, 0))
        detail_frame.pack_propagate(False)
        
        detail_label = tk.Label(
            detail_frame,
            text="Chi tiết",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="white"
        )
        detail_label.pack(fill=tk.X)
        
        self.detail_text = tk.Text(
            detail_frame,
            font=("Arial", 10),
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=10
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Avatar display in detail
        self.avatar_label = tk.Label(
            detail_frame,
            text="",
            bg="#ecf0f1"
        )
    
    def upload_image(self):
        """Upload search image"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh khuôn mặt",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_search_image(file_path)
            self.search_btn.config(state=tk.NORMAL)
            if self.status_callback:
                self.status_callback(f"Đã tải ảnh: {os.path.basename(file_path)}")
    
    def load_search_image(self, path):
        """Load and display search image"""
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh")
            return
        
        self.search_image = image.copy()
        
        # Resize for display
        display_image = self.resize_for_display(image, max_width=330, max_height=400)
        
        # Convert to PIL
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update label
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo
    
    def resize_for_display(self, image, max_width=330, max_height=400):
        """Resize image for display"""
        height, width = image.shape[:2]
        
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def search_students(self):
        """Search students by face"""
        if self.search_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước")
            return
        
        if self.status_callback:
            self.status_callback("Đang tìm kiếm...")
        
        try:
            # Search
            results = search_by_face_image(self.search_image, tolerance=0.6, max_results=10)
            
            self.search_results = results
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Display results
            if not results:
                messagebox.showinfo("Kết quả", "Không tìm thấy sinh viên khớp")
                if self.status_callback:
                    self.status_callback("Không tìm thấy kết quả")
                return
            
            # Add results to treeview
            for idx, result in enumerate(results):
                student = result['student']
                similarity = get_similarity_score(result['distance'])
                
                self.results_tree.insert(
                    "",
                    tk.END,
                    text=str(student['id']),
                    values=(
                        student.get('mssv', ''),
                        student.get('ho_ten', ''),
                        str(student.get('ngay_sinh', '')) if student.get('ngay_sinh') else '',
                        student.get('nien_khoa', ''),
                        f"{similarity}%"
                    ),
                    tags=('matched' if result['match'] else 'similar',)
                )
            
            # Configure tags
            self.results_tree.tag_configure('matched', background='#d4edda')
            self.results_tree.tag_configure('similar', background='#fff3cd')
            
            # Select first result
            first_item = self.results_tree.get_children()[0]
            self.results_tree.selection_set(first_item)
            self.results_tree.focus(first_item)
            self.on_result_select(None)
            
            if self.status_callback:
                self.status_callback(f"Tìm thấy {len(results)} kết quả")
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tìm kiếm: {str(e)}")
            if self.status_callback:
                self.status_callback("Lỗi khi tìm kiếm")
    
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

