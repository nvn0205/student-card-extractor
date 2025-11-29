"""Image preprocessing utilities using OpenCV"""
import cv2
import numpy as np


def resize_image(image, max_width=1920, max_height=1080):
    """
    Resize image nếu quá lớn, giữ nguyên tỷ lệ
    
    Args:
        image: numpy array (OpenCV image)
        max_width: int
        max_height: int
    
    Returns:
        numpy array: Resized image
    """
    height, width = image.shape[:2]
    
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


def normalize_image(image):
    """
    Normalize ảnh (convert to grayscale nếu cần)
    
    Args:
        image: numpy array
    
    Returns:
        numpy array: Grayscale image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return gray


def enhance_contrast(image, alpha=1.5, beta=0):
    """
    Tăng độ tương phản của ảnh
    
    Args:
        image: numpy array (grayscale)
        alpha: float - contrast control (1.0-3.0)
        beta: int - brightness control
    
    Returns:
        numpy array: Enhanced image
    """
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced


def denoise_image(image, method='gaussian'):
    """
    Loại bỏ noise từ ảnh
    
    Args:
        image: numpy array
        method: str - 'gaussian', 'bilateral', 'median'
    
    Returns:
        numpy array: Denoised image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    else:
        return image


def adjust_brightness(image, value):
    """
    Điều chỉnh độ sáng
    
    Args:
        image: numpy array
        value: int - giá trị điều chỉnh (-100 đến 100)
    
    Returns:
        numpy array: Adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image


def sharpen_image(image):
    """
    Làm sắc nét ảnh
    
    Args:
        image: numpy array
    
    Returns:
        numpy array: Sharpened image
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def preprocess_for_ocr(image):
    """
    Tiền xử lý ảnh tối ưu cho OCR
    
    Args:
        image: numpy array
    
    Returns:
        numpy array: Preprocessed image
    """
    # Resize nếu quá lớn (giữ nguyên tỷ lệ, chỉ resize nếu quá lớn)
    image = resize_image(image, max_width=1920, max_height=1080)
    
    # Convert to grayscale
    gray = normalize_image(image)
    
    # Thử nhiều phương pháp và chọn cái tốt nhất
    
    # Method 1: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 2: OTSU threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 3: Morphological operations để làm sạch
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    
    # Method 4: Enhance contrast rồi threshold
    enhanced = enhance_contrast(gray, alpha=1.5)
    denoised = denoise_image(enhanced, method='bilateral')
    _, threshold_enhanced = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ưu tiên adaptive threshold vì thường tốt nhất cho text
    return morph


def preprocess_for_detection(image):
    """
    Tiền xử lý ảnh cho việc phát hiện thẻ (detection)
    
    Args:
        image: numpy array
    
    Returns:
        numpy array: Preprocessed image
    """
    # Resize
    image = resize_image(image)
    
    # Convert to grayscale
    gray = normalize_image(image)
    
    # Enhance contrast
    enhanced = enhance_contrast(gray)
    
    # Denoise
    denoised = denoise_image(enhanced)
    
    return denoised

