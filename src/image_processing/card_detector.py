"""Card detection and extraction using OpenCV"""
import cv2
import numpy as np
from .preprocessor import preprocess_for_detection


def find_card_contour(image):
    """
    Tìm contour của thẻ trong ảnh
    
    Args:
        image: numpy array (grayscale)
    
    Returns:
        numpy array: Contour của thẻ hoặc None
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilation để nối các đường viền
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Tìm contour có hình chữ nhật (thẻ thường là hình chữ nhật)
    for contour in contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Nếu có 4 điểm thì có thể là hình chữ nhật
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            # Kiểm tra diện tích đủ lớn (ít nhất 10% diện tích ảnh)
            img_area = image.shape[0] * image.shape[1]
            if area > img_area * 0.1:
                return approx
    
    # Nếu không tìm thấy hình chữ nhật, trả về contour lớn nhất
    if contours:
        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.02 * peri, True)
        if len(approx) >= 4:
            return approx
    
    return None


def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: numpy array shape (4, 2)
    
    Returns:
        numpy array: Ordered points
    """
    # Khởi tạo mảng kết quả
    rect = np.zeros((4, 2), dtype="float32")
    
    # Tổng nhỏ nhất là top-left, tổng lớn nhất là bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Hiệu nhỏ nhất là top-right, hiệu lớn nhất là bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def four_point_transform(image, pts):
    """
    Áp dụng perspective transform để crop thẻ
    
    Args:
        image: numpy array
        pts: numpy array shape (4, 2) - 4 góc của thẻ
    
    Returns:
        numpy array: Cropped và straightened card image
    """
    # Sắp xếp các điểm
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính chiều rộng và chiều cao của thẻ
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Điểm đích sau khi transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Tính matrix transform
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def detect_and_extract_card(image):
    """
    Phát hiện và trích xuất thẻ từ ảnh
    
    Args:
        image: numpy array (BGR image)
    
    Returns:
        tuple: (card_image, success_flag)
            - card_image: numpy array - ảnh thẻ đã được crop và thẳng
            - success_flag: bool - True nếu tìm thấy thẻ
    """
    # Tiền xử lý
    processed = preprocess_for_detection(image)
    
    # Tìm contour
    contour = find_card_contour(processed)
    
    if contour is None:
        return image, False
    
    # Reshape contour về (4, 2)
    if len(contour) == 4:
        pts = contour.reshape(4, 2)
    else:
        # Nếu không có đúng 4 điểm, thử lấy 4 góc của bounding box
        x, y, w, h = cv2.boundingRect(contour)
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype="float32")
    
    # Transform để crop thẻ
    card_image = four_point_transform(image, pts)
    
    return card_image, True


def detect_card_simple(image):
    """
    Phương pháp đơn giản hơn: chỉ crop phần trung tâm nếu không detect được
    
    Args:
        image: numpy array
    
    Returns:
        numpy array: Card image
    """
    card_image, success = detect_and_extract_card(image)
    
    if not success:
        # Nếu không detect được, trả về toàn bộ ảnh
        return image
    
    return card_image

