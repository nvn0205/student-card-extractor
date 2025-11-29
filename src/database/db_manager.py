"""Database connection manager"""
import mysql.connector
from mysql.connector import Error
import sys
import os

# Add config directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from config.database import DB_CONFIG


class DBManager:
    """Manages MySQL database connections"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**DB_CONFIG)
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                return True
        except Error as e:
            print(f"Lỗi kết nối database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            if self.cursor:
                self.cursor.close()
            self.connection.close()
    
    def execute_query(self, query, params=None):
        """Execute SELECT query"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            return self.cursor.fetchall()
        except Error as e:
            print(f"Lỗi thực thi query: {e}")
            return None
    
    def execute_update(self, query, params=None):
        """Execute INSERT/UPDATE/DELETE query"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            self.connection.commit()
            return self.cursor.rowcount
        except Error as e:
            if self.connection:
                self.connection.rollback()
            print(f"Lỗi thực thi update: {e}")
            return 0
    
    def execute_insert(self, query, params=None):
        """Execute INSERT query and return last insert id"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            self.connection.commit()
            return self.cursor.lastrowid
        except Error as e:
            if self.connection:
                self.connection.rollback()
            print(f"Lỗi thực thi insert: {e}")
            return None
    
    def get_connection(self):
        """Get current connection"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        return self.connection
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Global instance
db_manager = DBManager()

