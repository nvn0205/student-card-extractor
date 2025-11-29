"""Face extraction from student card"""
import cv2
import face_recognition
import numpy as np


def extract_face_region(card_image, padding=20):
    """
    Trích xuất vùng chứa khuôn mặt từ ảnh thẻ
    
    Args:
        card_image: numpy array (BGR image)
        padding: int - padding xung quanh khuôn mặt (pixels)
    
    Returns:
        tuple: (face_image, face_locations)
            - face_image: numpy array - ảnh chân dung đã crop
            - face_locations: list - vị trí khuôn mặt (top, right, bottom, left)
    """
    # Resize ảnh nếu quá nhỏ (face_recognition hoạt động tốt hơn với ảnh lớn hơn)
    h, w = card_image.shape[:2]
    scale_factor = 1.0
    
    # Nếu ảnh nhỏ hơn 500px, resize lên
    if max(h, w) < 500:
        scale_factor = 500.0 / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        resized_image = cv2.resize(card_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"  Resized image from {card_image.shape} to {resized_image.shape} for better face detection")
    else:
        resized_image = card_image
    
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Thử detect với HOG model trước (nhanh)
    face_locations = face_recognition.face_locations(rgb_image, model='hog', number_of_times_to_upsample=1)
    
    # Nếu không tìm thấy, thử với CNN model (chính xác hơn nhưng chậm)
    if not face_locations:
        print("  Trying CNN model for face detection...")
        try:
            face_locations = face_recognition.face_locations(rgb_image, model='cnn', number_of_times_to_upsample=0)
        except Exception as e:
            print(f"  CNN model failed: {e}, continuing...")
    
    if not face_locations:
        print(f"  No faces detected. Image shape: {resized_image.shape}")
        return None, None
    
    print(f"  Found {len(face_locations)} face(s)")
    
    # Lấy khuôn mặt lớn nhất (thường là khuôn mặt chính trong thẻ)
    # Sắp xếp theo diện tích
    face_areas = [(bottom - top) * (right - left) for (top, right, bottom, left) in face_locations]
    largest_face_idx = np.argmax(face_areas)
    
    top, right, bottom, left = face_locations[largest_face_idx]
    print(f"  Largest face location: top={top}, right={right}, bottom={bottom}, left={left}")
    
    # Scale lại về kích thước gốc nếu đã resize
    if scale_factor != 1.0:
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)
    
    # Thêm padding
    h, w = card_image.shape[:2]
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(h, bottom + padding)
    right = min(w, right + padding)
    
    # Crop ảnh khuôn mặt từ ảnh gốc (không resize)
    face_image = card_image[top:bottom, left:right]
    
    # Kiểm tra kích thước ảnh sau khi crop
    if face_image.size == 0:
        print("  Warning: Cropped face image is empty!")
        return None, None
    
    print(f"  Face image cropped: {face_image.shape}")
    
    return face_image, (top, right, bottom, left)


def extract_and_save_face(card_image, output_path, padding=20):
    """
    Trích xuất và lưu ảnh chân dung
    
    Args:
        card_image: numpy array
        output_path: str - đường dẫn lưu ảnh
        padding: int
    
    Returns:
        tuple: (success, face_image_path, face_location)
    """
    face_image, face_location = extract_face_region(card_image, padding)
    
    if face_image is None:
        return False, None, None
    
    # Lưu ảnh
    cv2.imwrite(output_path, face_image)
    
    return True, output_path, face_location


def get_face_encoding_from_card(card_image):
    """
    Lấy face encoding từ ảnh thẻ (dùng cho face recognition)
    
    Args:
        card_image: numpy array (BGR)
    
    Returns:
        numpy array: Face encoding hoặc None
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image, model='hog')
    
    if not face_locations:
        return None
    
    # Lấy encoding của khuôn mặt đầu tiên
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if face_encodings:
        return face_encodings[0]
    
    return None


def extract_face_with_resize(card_image, target_size=(200, 250), padding=20):
    """
    Trích xuất ảnh chân dung và resize về kích thước cố định
    
    Args:
        card_image: numpy array
        target_size: tuple (width, height)
        padding: int
    
    Returns:
        numpy array: Resized face image hoặc None
    """
    face_image, _ = extract_face_region(card_image, padding)
    
    if face_image is None:
        return None
    
    # Resize về kích thước cố định
    face_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
    
    return face_resized

