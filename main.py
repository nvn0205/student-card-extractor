"""Main entry point for Student Card OCR System"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run main
from src.gui.main_window import MainWindow
import tkinter as tk

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

