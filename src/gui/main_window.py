"""Main GUI window - Modern Design"""
import tkinter as tk
from tkinter import ttk, messagebox
from .extract_window import ExtractWindow
from .search_window import SearchWindow
from ..database.student_dao import StudentDAO


class MainWindow:
    """Main application window with modern design"""
    
    # Modern color palette
    COLORS = {
        'primary': '#6366f1',      # Indigo
        'secondary': '#8b5cf6',    # Purple
        'success': '#10b981',      # Green
        'warning': '#f59e0b',      # Amber
        'danger': '#ef4444',       # Red
        'dark': '#1f2937',         # Dark gray
        'light': '#f9fafb',        # Light gray
        'white': '#ffffff',
        'bg': '#f3f4f6',
        'card_bg': '#ffffff',
        'text': '#111827',
        'text_light': '#6b7280',
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Hệ thống Trích xuất Thông tin Thẻ Sinh viên")
        self.root.geometry("900x700")
        self.root.configure(bg=self.COLORS['bg'])
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
        """Create modern UI widgets"""
        # Header with gradient effect simulation
        header_frame = tk.Frame(
            self.root, 
            bg=self.COLORS['primary'],
            height=120
        )
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title section
        title_container = tk.Frame(header_frame, bg=self.COLORS['primary'])
        title_container.pack(expand=True)
        
        icon_label = tk.Label(
            title_container,
            text="📸",
            font=("Segoe UI Emoji", 32),
            bg=self.COLORS['primary'],
            fg=self.COLORS['white']
        )
        icon_label.pack()
        
        title_label = tk.Label(
            title_container,
            text="Hệ thống Trích xuất Thông tin",
            font=("Segoe UI", 24, "bold"),
            bg=self.COLORS['primary'],
            fg=self.COLORS['white']
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_container,
            text="Thẻ Sinh viên - Student Card OCR System",
            font=("Segoe UI", 11),
            bg=self.COLORS['primary'],
            fg="#e0e7ff"  # Light indigo
        )
        subtitle_label.pack()
        
        # Main content area
        content_frame = tk.Frame(self.root, bg=self.COLORS['bg'], padx=40, pady=40)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Welcome text
        welcome_label = tk.Label(
            content_frame,
            text="Chào mừng bạn đến với hệ thống! 👋",
            font=("Segoe UI", 14),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text_light']
        )
        welcome_label.pack(pady=(0, 30))
        
        # Action cards container
        cards_frame = tk.Frame(content_frame, bg=self.COLORS['bg'])
        cards_frame.pack(expand=True)
        
        # Card 1: Extract Information
        extract_card = self.create_action_card(
            cards_frame,
            icon="📄",
            title="Trích xuất thông tin",
            description="Tải lên ảnh thẻ sinh viên\nvà trích xuất thông tin tự động",
            color=self.COLORS['primary'],
            command=self.open_extract_window
        )
        extract_card.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Card 2: Face Search
        search_card = self.create_action_card(
            cards_frame,
            icon="🔍",
            title="Tìm kiếm theo khuôn mặt",
            description="Tìm kiếm sinh viên bằng\nảnh khuôn mặt (Camera realtime)",
            color=self.COLORS['success'],
            command=self.open_search_window
        )
        search_card.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Card 3: View List
        view_card = self.create_action_card(
            cards_frame,
            icon="📋",
            title="Danh sách sinh viên",
            description="Xem toàn bộ sinh viên\nđã được lưu trong hệ thống",
            color=self.COLORS['warning'],
            command=self.view_all_students
        )
        view_card.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Status bar
        status_frame = tk.Frame(
            self.root, 
            bg=self.COLORS['dark'],
            height=35
        )
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        status_container = tk.Frame(status_frame, bg=self.COLORS['dark'])
        status_container.pack(expand=True)
        
        status_icon = tk.Label(
            status_container,
            text="●",
            font=("Arial", 10),
            bg=self.COLORS['dark'],
            fg=self.COLORS['success']
        )
        status_icon.pack(side=tk.LEFT, padx=(10, 5))
        
        self.status_label = tk.Label(
            status_container,
            text="Sẵn sàng",
            font=("Segoe UI", 10),
            bg=self.COLORS['dark'],
            fg=self.COLORS['white']
        )
        self.status_label.pack(side=tk.LEFT)
    
    def create_action_card(self, parent, icon, title, description, color, command):
        """Create a modern action card"""
        # Card container
        card = tk.Frame(
            parent,
            bg=self.COLORS['card_bg'],
            relief=tk.FLAT,
            width=220,
            height=280
        )
        card.pack_propagate(False)
        
        # Card shadow effect (using border)
        card_border = tk.Frame(
            card,
            bg='#e5e7eb',
            padx=2,
            pady=2
        )
        card_border.pack(fill=tk.BOTH, expand=True)
        
        # Card content
        card_content = tk.Frame(card_border, bg=self.COLORS['card_bg'])
        card_content.pack(fill=tk.BOTH, expand=True)
        
        # Icon
        icon_label = tk.Label(
            card_content,
            text=icon,
            font=("Segoe UI Emoji", 48),
            bg=self.COLORS['card_bg'],
            fg=color
        )
        icon_label.pack(pady=(20, 10))
        
        # Title
        title_label = tk.Label(
            card_content,
            text=title,
            font=("Segoe UI", 14, "bold"),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['text'],
            wraplength=180
        )
        title_label.pack(pady=(0, 10))
        
        # Description
        desc_label = tk.Label(
            card_content,
            text=description,
            font=("Segoe UI", 10),
            bg=self.COLORS['card_bg'],
            fg=self.COLORS['text_light'],
            wraplength=180,
            justify=tk.CENTER
        )
        desc_label.pack(pady=(0, 20))
        
        # Action button
        action_btn = tk.Button(
            card_content,
            text="Bắt đầu →",
            font=("Segoe UI", 11, "bold"),
            bg=color,
            fg=self.COLORS['white'],
            relief=tk.FLAT,
            cursor="hand2",
            command=command,
            padx=20,
            pady=8,
            activebackground=color,
            activeforeground=self.COLORS['white']
        )
        action_btn.pack(pady=(0, 20))
        
        # Hover effect simulation
        def on_enter(e):
            card_border.config(bg='#d1d5db')
            action_btn.config(bg=color, cursor="hand2")
        
        def on_leave(e):
            card_border.config(bg='#e5e7eb')
            action_btn.config(bg=color)
        
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        card_content.bind("<Enter>", on_enter)
        card_content.bind("<Leave>", on_leave)
        
        return card
    
    def open_extract_window(self):
        """Open extract window"""
        extract_window = tk.Toplevel(self.root)
        ExtractWindow(extract_window, self.update_status)
    
    def open_search_window(self):
        """Open search window"""
        search_window = tk.Toplevel(self.root)
        SearchWindow(search_window, self.update_status)
    
    def view_all_students(self):
        """View all students in a modern window"""
        view_window = tk.Toplevel(self.root)
        view_window.title("📋 Danh sách Sinh viên")
        view_window.geometry("1000x650")
        view_window.configure(bg=self.COLORS['bg'])
        
        # Header
        header = tk.Frame(
            view_window,
            bg=self.COLORS['primary'],
            height=80
        )
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        header_content = tk.Frame(header, bg=self.COLORS['primary'])
        header_content.pack(expand=True)
        
        header_title = tk.Label(
            header_content,
            text="📋 Danh sách Sinh viên",
            font=("Segoe UI", 18, "bold"),
            bg=self.COLORS['primary'],
            fg=self.COLORS['white']
        )
        header_title.pack(pady=20)
        
        # Content area
        content = tk.Frame(view_window, bg=self.COLORS['bg'], padx=20, pady=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Card container
        card_container = tk.Frame(content, bg=self.COLORS['card_bg'], padx=10, pady=10)
        card_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(card_container, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(card_container, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Modern Treeview
        tree = ttk.Treeview(
            card_container,
            columns=("MSSV", "Họ tên", "Ngày sinh", "Niên khóa", "Ngày hết hạn"),
            show="headings",
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            style="Modern.Treeview"
        )
        
        # Configure style
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
        tree.heading("#0", text="ID")
        tree.heading("MSSV", text="Mã SV")
        tree.heading("Họ tên", text="Họ và Tên")
        tree.heading("Ngày sinh", text="Ngày Sinh")
        tree.heading("Niên khóa", text="Niên Khóa")
        tree.heading("Ngày hết hạn", text="Thẻ có giá trị đến")
        
        tree.column("#0", width=50, anchor=tk.CENTER)
        tree.column("MSSV", width=120, anchor=tk.CENTER)
        tree.column("Họ tên", width=220, anchor=tk.W)
        tree.column("Ngày sinh", width=130, anchor=tk.CENTER)
        tree.column("Niên khóa", width=130, anchor=tk.CENTER)
        tree.column("Ngày hết hạn", width=150, anchor=tk.CENTER)
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=tree.yview)
        h_scrollbar.config(command=tree.xview)
        
        # Load data
        try:
            students = StudentDAO.get_all()
            if not students:
                empty_label = tk.Label(
                    content,
                    text="📭 Chưa có sinh viên nào trong hệ thống",
                    font=("Segoe UI", 12),
                    bg=self.COLORS['bg'],
                    fg=self.COLORS['text_light']
                )
                empty_label.pack(expand=True)
            else:
                for idx, student in enumerate(students):
                    tag = "evenrow" if idx % 2 == 0 else "oddrow"
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
                        ),
                        tags=(tag,)
                    )
                
                # Configure row colors
                tree.tag_configure("evenrow", background=self.COLORS['white'])
                tree.tag_configure("oddrow", background=self.COLORS['light'])
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải dữ liệu: {str(e)}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
