"""Main GUI window"""
import tkinter as tk
from tkinter import ttk, messagebox
from .extract_window import ExtractWindow
from .search_window import SearchWindow
from ..database.student_dao import StudentDAO


class MainWindow:
    """Main application window"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống trích xuất thông tin thẻ sinh viên")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Center window
        self.center_window()
        
        # Create UI
        self.create_widgets()
        
        # Connect to database
        try:
            from ..database.db_manager import db_manager
            if not db_manager.connect():
                messagebox.showerror(
                    "Lỗi kết nối",
                    "Không thể kết nối đến database. Vui lòng kiểm tra cấu hình."
                )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khởi tạo: {str(e)}")
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Create UI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Hệ thống trích xuất thông tin thẻ sinh viên",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=25)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg="#ecf0f1", padx=50, pady=50)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        buttons_frame = tk.Frame(content_frame, bg="#ecf0f1")
        buttons_frame.pack(expand=True)
        
        # Extract button
        extract_btn = tk.Button(
            buttons_frame,
            text="Trích xuất thông tin",
            font=("Arial", 14),
            bg="#3498db",
            fg="white",
            width=25,
            height=3,
            cursor="hand2",
            command=self.open_extract_window,
            relief=tk.FLAT
        )
        extract_btn.pack(pady=20)
        
        # Search button
        search_btn = tk.Button(
            buttons_frame,
            text="Tìm kiếm theo ảnh khuôn mặt",
            font=("Arial", 14),
            bg="#27ae60",
            fg="white",
            width=25,
            height=3,
            cursor="hand2",
            command=self.open_search_window,
            relief=tk.FLAT
        )
        search_btn.pack(pady=20)
        
        # View All button
        view_all_btn = tk.Button(
            buttons_frame,
            text="Xem danh sách sinh viên",
            font=("Arial", 14),
            bg="#e67e22",
            fg="white",
            width=25,
            height=3,
            cursor="hand2",
            command=self.view_all_students,
            relief=tk.FLAT
        )
        view_all_btn.pack(pady=20)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Sẵn sàng",
            font=("Arial", 10),
            bg="#34495e",
            fg="white"
        )
        self.status_label.pack(pady=5)
    
    def open_extract_window(self):
        """Open extract window"""
        extract_window = tk.Toplevel(self.root)
        ExtractWindow(extract_window, self.update_status)
    
    def open_search_window(self):
        """Open search window"""
        search_window = tk.Toplevel(self.root)
        SearchWindow(search_window, self.update_status)
    
    def view_all_students(self):
        """View all students in a new window"""
        view_window = tk.Toplevel(self.root)
        view_window.title("Danh sách sinh viên")
        view_window.geometry("900x600")
        
        # Create treeview
        tree_frame = tk.Frame(view_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        tree = ttk.Treeview(
            tree_frame,
            columns=("MSSV", "Họ tên", "Ngày sinh", "Niên khóa", "Ngày hết hạn"),
            show="headings",
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        # Configure columns
        tree.heading("#0", text="ID")
        tree.heading("MSSV", text="MSSV")
        tree.heading("Họ tên", text="Họ tên")
        tree.heading("Ngày sinh", text="Ngày sinh")
        tree.heading("Niên khóa", text="Niên khóa")
        tree.heading("Ngày hết hạn", text="Ngày hết hạn")
        
        tree.column("#0", width=40)
        tree.column("MSSV", width=120)
        tree.column("Họ tên", width=200)
        tree.column("Ngày sinh", width=120)
        tree.column("Niên khóa", width=120)
        tree.column("Ngày hết hạn", width=140)
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=tree.yview)
        h_scrollbar.config(command=tree.xview)
        
        # Load data
        try:
            students = StudentDAO.get_all()
            for student in students:
                tree.insert(
                    "",
                    tk.END,
                    text=str(student['id']),
                    values=(
                        student.get('mssv', ''),
                        student.get('ho_ten', ''),
                        str(student.get('ngay_sinh', '')) if student.get('ngay_sinh') else '',
                        student.get('nien_khoa', ''),
                        str(student.get('ngay_het_han', '')) if student.get('ngay_het_han') else ''
                    )
                )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải dữ liệu: {str(e)}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

