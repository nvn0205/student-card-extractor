"""Face extraction from student card"""
import cv2
import face_recognition
import numpy as np


def extract_face_region(card_image, padding=20):
    """
    Trích xuất vùng chứa khuôn mặt từ ảnh thẻ (bao gồm cả đầu và cổ)
    
    Args:
        card_image: numpy array (BGR image)
        padding: int - padding cơ bản xung quanh khuôn mặt (pixels)
                  Sẽ được tính toán động dựa trên kích thước khuôn mặt
    
    Returns:
        tuple: (face_image, face_locations)
            - face_image: numpy array - ảnh chân dung đã crop (toàn bộ đầu và cổ)
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
        resized_image = card_image.copy()
    
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
    print(f"  Largest face location (on resized): top={top}, right={right}, bottom={bottom}, left={left}")
    
    # Tính toán padding động dựa trên kích thước khuôn mặt
    face_height = bottom - top
    face_width = right - left
    
    # Padding theo % kích thước khuôn mặt để đảm bảo lấy đủ phần
    # Trên và dưới: thêm nhiều hơn để lấy cả đầu và cổ
    padding_top_pct = 0.8  # 80% chiều cao khuôn mặt phía trên (lấy phần đầu)
    padding_bottom_pct = 0.6  # 60% chiều cao khuôn mặt phía dưới (lấy phần cổ)
    padding_side_pct = 0.3  # 30% chiều rộng khuôn mặt mỗi bên
    
    # Tính padding động
    dynamic_padding_top = int(face_height * padding_top_pct)
    dynamic_padding_bottom = int(face_height * padding_bottom_pct)
    dynamic_padding_left = int(face_width * padding_side_pct)
    dynamic_padding_right = int(face_width * padding_side_pct)
    
    # Kết hợp với padding cố định, lấy giá trị lớn hơn
    padding_top = max(dynamic_padding_top, padding * 2)
    padding_bottom = max(dynamic_padding_bottom, padding * 2)
    padding_left = max(dynamic_padding_left, padding)
    padding_right = max(dynamic_padding_right, padding)
    
    print(f"  Dynamic padding - top: {padding_top}, bottom: {padding_bottom}, left: {padding_left}, right: {padding_right}")
    
    # Thêm padding trên ảnh đã resize
    resized_h, resized_w = resized_image.shape[:2]
    top_padded = max(0, top - padding_top)
    left_padded = max(0, left - padding_left)
    bottom_padded = min(resized_h, bottom + padding_bottom)
    right_padded = min(resized_w, right + padding_right)
    
    print(f"  Padded location (on resized): top={top_padded}, right={right_padded}, bottom={bottom_padded}, left={left_padded}")
    
    # Scale lại về kích thước gốc
    if scale_factor != 1.0:
        # Tính lại padding trên ảnh gốc
        face_height_original = int(face_height / scale_factor)
        face_width_original = int(face_width / scale_factor)
        
        padding_top_original = int(face_height_original * padding_top_pct)
        padding_bottom_original = int(face_height_original * padding_bottom_pct)
        padding_left_original = int(face_width_original * padding_side_pct)
        padding_right_original = int(face_width_original * padding_side_pct)
        
        # Scale tọa độ về ảnh gốc
        top_original = int(top / scale_factor)
        right_original = int(right / scale_factor)
        bottom_original = int(bottom / scale_factor)
        left_original = int(left / scale_factor)
        
        # Áp dụng padding trên ảnh gốc
        top_final = max(0, top_original - max(padding_top_original, padding * 2))
        left_final = max(0, left_original - max(padding_left_original, padding))
        bottom_final = min(h, bottom_original + max(padding_bottom_original, padding * 2))
        right_final = min(w, right_original + max(padding_right_original, padding))
    else:
        # Không cần scale, dùng trực tiếp tọa độ đã padding
        top_final = top_padded
        left_final = left_padded
        bottom_final = bottom_padded
        right_final = right_padded
    
    print(f"  Final crop location (on original): top={top_final}, right={right_final}, bottom={bottom_final}, left={left_final}")
    
    # Crop ảnh khuôn mặt từ ảnh gốc (bao gồm cả đầu và cổ)
    face_image = card_image[top_final:bottom_final, left_final:right_final]
    
    # Kiểm tra kích thước ảnh sau khi crop
    if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
        print("  Warning: Cropped face image is empty!")
        return None, None
    
    print(f"  Face image cropped (full head + neck): {face_image.shape}")
    print(f"  Crop ratio: {face_image.shape[1]/face_image.shape[0]:.2f} (width/height)")
    
    return face_image, (top_final, right_final, bottom_final, left_final)


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

