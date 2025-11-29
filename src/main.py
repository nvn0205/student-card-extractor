"""Main entry point for Student Card OCR System"""
import tkinter as tk
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now we can import from src package
from src.gui.main_window import MainWindow


def main():
    """Main function"""
    # Create root window
    root = tk.Tk()
    
    # Create and run main window
    app = MainWindow(root)
    
    # Run application
    root.mainloop()


if __name__ == "__main__":
    main()

