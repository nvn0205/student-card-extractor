"""Face recognition and matching module"""
import face_recognition
import numpy as np
import cv2
from ..database.student_dao import StudentDAO


def encode_face(image):
    """
    Encode khuôn mặt từ ảnh thành feature vector
    
    Args:
        image: numpy array (BGR image)
    
    Returns:
        numpy array: Face encoding (128-dimensional vector) hoặc None
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image, model='hog')
    
    if not face_locations:
        return None
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if face_encodings:
        return face_encodings[0]
    
    return None


def compare_faces(known_encoding, face_encoding_to_check, tolerance=0.6):
    """
    So sánh 2 face encodings
    
    Args:
        known_encoding: numpy array - encoding đã biết
        face_encoding_to_check: numpy array - encoding cần kiểm tra
        tolerance: float - ngưỡng so sánh (thấp hơn = chặt chẽ hơn)
    
    Returns:
        tuple: (match, distance)
            - match: bool - True nếu match
            - distance: float - khoảng cách euclidean
    """
    # Tính khoảng cách
    distance = face_recognition.face_distance([known_encoding], face_encoding_to_check)[0]
    
    # So sánh với tolerance
    match = distance <= tolerance
    
    return match, distance


def find_matching_students(query_face_encoding, tolerance=0.5, max_results=5):
    """
    Tìm kiếm sinh viên khớp với face encoding
    Chỉ trả về những kết quả thực sự match hoặc top kết quả tốt nhất
    
    Args:
        query_face_encoding: numpy array - face encoding cần tìm
        tolerance: float - ngưỡng so sánh (mặc định 0.5 - chặt chẽ hơn)
        max_results: int - số kết quả tối đa (mặc định 5)
    
    Returns:
        list: List of dicts với keys:
            - student: dict - thông tin sinh viên
            - distance: float - khoảng cách
            - match: bool - True nếu match
    """
    if query_face_encoding is None:
        return []
    
    # Lấy tất cả students có face encoding
    students = StudentDAO.get_all_with_encodings()
    
    matched_results = []  # Kết quả match thực sự
    all_results = []      # Tất cả kết quả để so sánh
    
    for student in students:
        if student.get('face_encoding') is None:
            continue
        
        # So sánh
        match, distance = compare_faces(
            student['face_encoding'],
            query_face_encoding,
            tolerance
        )
        
        result_item = {
            'student': student,
            'distance': distance,
            'match': match
        }
        
        # Thêm vào all_results để sort
        all_results.append(result_item)
        
        # Chỉ thêm vào matched_results nếu thực sự match
        if match:
            matched_results.append(result_item)
    
    # Sắp xếp theo distance (tăng dần - càng nhỏ càng giống)
    matched_results.sort(key=lambda x: x['distance'])
    all_results.sort(key=lambda x: x['distance'])
    
    # Ưu tiên: trả về các kết quả match thực sự
    if matched_results:
        return matched_results[:max_results]
    
    # Nếu không có kết quả match, chỉ trả về top 1-2 kết quả tốt nhất (nếu distance < 0.7)
    if all_results and all_results[0]['distance'] < 0.7:
        return all_results[:min(2, max_results)]
    
    # Không có kết quả khả thi
    return []


def search_by_face_image(image, tolerance=0.6, max_results=10):
    """
    Tìm kiếm sinh viên theo ảnh khuôn mặt
    
    Args:
        image: numpy array (BGR image)
        tolerance: float - ngưỡng so sánh
        max_results: int - số kết quả tối đa
    
    Returns:
        list: List of matching students với distance và match flag
    """
    # Encode face từ ảnh
    face_encoding = encode_face(image)
    
    if face_encoding is None:
        return []
    
    # Tìm kiếm
    return find_matching_students(face_encoding, tolerance, max_results)


def get_similarity_score(distance):
    """
    Chuyển đổi distance thành similarity score (0-100%)
    
    Args:
        distance: float - face distance
    
    Returns:
        float: Similarity score (0-100)
    """
    # Distance thường từ 0 (giống nhất) đến 1+ (khác nhau)
    # Chuyển thành similarity: 100% khi distance = 0, giảm dần
    similarity = max(0, (1 - distance) * 100)
    return round(similarity, 2)

