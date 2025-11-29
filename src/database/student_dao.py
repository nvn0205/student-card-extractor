"""Data Access Object for Student operations"""
import pickle
import numpy as np
from datetime import datetime
from .db_manager import db_manager


class StudentDAO:
    """DAO for student CRUD operations"""
    
    @staticmethod
    def create(student_data):
        """
        Insert new student into database
        
        Args:
            student_data: dict với keys: mssv, ho_ten, ngay_sinh, nien_khoa, ngay_het_han,
                         avatar_path, face_encoding (numpy array)
        
        Returns:
            int: ID của student mới tạo, None nếu lỗi
        """
        query = """
            INSERT INTO students (mssv, ho_ten, ngay_sinh, nien_khoa, ngay_het_han, avatar_path, face_encoding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Serialize face encoding
        face_encoding_bytes = None
        if student_data.get('face_encoding') is not None:
            face_encoding_bytes = pickle.dumps(student_data['face_encoding'])
        
        params = (
            student_data.get('mssv'),
            student_data.get('ho_ten'),
            student_data.get('ngay_sinh'),
            student_data.get('nien_khoa'),
            student_data.get('ngay_het_han'),
            student_data.get('avatar_path'),
            face_encoding_bytes
        )
        
        return db_manager.execute_insert(query, params)
    
    @staticmethod
    def get_by_id(student_id):
        """
        Get student by ID
        
        Args:
            student_id: int
        
        Returns:
            dict: Student data hoặc None
        """
        query = "SELECT * FROM students WHERE id = %s"
        results = db_manager.execute_query(query, (student_id,))
        
        if results and len(results) > 0:
            student = results[0]
            # Deserialize face encoding
            if student['face_encoding']:
                student['face_encoding'] = pickle.loads(student['face_encoding'])
            return student
        return None
    
    @staticmethod
    def get_by_mssv(mssv):
        """
        Get student by MSSV
        
        Args:
            mssv: string
        
        Returns:
            dict: Student data hoặc None
        """
        query = "SELECT * FROM students WHERE mssv = %s"
        results = db_manager.execute_query(query, (mssv,))
        
        if results and len(results) > 0:
            student = results[0]
            # Deserialize face encoding
            if student['face_encoding']:
                student['face_encoding'] = pickle.loads(student['face_encoding'])
            return student
        return None
    
    @staticmethod
    def get_all():
        """
        Get all students
        
        Returns:
            list: List of student dicts
        """
        query = "SELECT id, mssv, ho_ten, ngay_sinh, nien_khoa, ngay_het_han, avatar_path, created_at FROM students ORDER BY created_at DESC"
        results = db_manager.execute_query(query)
        return results or []
    
    @staticmethod
    def get_all_with_encodings():
        """
        Get all students with face encodings (for face matching)
        
        Returns:
            list: List of student dicts with face_encoding
        """
        query = "SELECT * FROM students WHERE face_encoding IS NOT NULL"
        results = db_manager.execute_query(query)
        
        if results:
            for student in results:
                if student['face_encoding']:
                    student['face_encoding'] = pickle.loads(student['face_encoding'])
            return results
        return []
    
    @staticmethod
    def update(student_id, student_data):
        """
        Update student information
        
        Args:
            student_id: int
            student_data: dict với các fields cần update
        
        Returns:
            int: Số rows được update
        """
        fields = []
        params = []
        
        for key in ['mssv', 'ho_ten', 'ngay_sinh', 'nien_khoa', 'ngay_het_han', 'avatar_path']:
            if key in student_data:
                fields.append(f"{key} = %s")
                params.append(student_data[key])
        
        # Handle face encoding separately
        if 'face_encoding' in student_data:
            fields.append("face_encoding = %s")
            face_encoding_bytes = None
            if student_data['face_encoding'] is not None:
                face_encoding_bytes = pickle.dumps(student_data['face_encoding'])
            params.append(face_encoding_bytes)
        
        if not fields:
            return 0
        
        params.append(student_id)
        query = f"UPDATE students SET {', '.join(fields)} WHERE id = %s"
        
        return db_manager.execute_update(query, tuple(params))
    
    @staticmethod
    def delete(student_id):
        """
        Delete student by ID
        
        Args:
            student_id: int
        
        Returns:
            int: Số rows được delete
        """
        query = "DELETE FROM students WHERE id = %s"
        return db_manager.execute_update(query, (student_id,))
    
    @staticmethod
    def search_by_name(keyword):
        """
        Search students by name
        
        Args:
            keyword: string
        
        Returns:
            list: List of matching students
        """
        query = "SELECT id, mssv, ho_ten, ngay_sinh, nien_khoa, ngay_het_han, avatar_path, created_at FROM students WHERE ho_ten LIKE %s ORDER BY ho_ten"
        results = db_manager.execute_query(query, (f'%{keyword}%',))
        return results or []

