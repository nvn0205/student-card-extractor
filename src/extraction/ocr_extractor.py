"""OCR extraction using Tesseract"""
import pytesseract
import cv2
import re
import os
import platform
import shutil
from datetime import datetime
from ..image_processing.preprocessor import preprocess_for_ocr, enhance_contrast, normalize_image

# Cấu hình đường dẫn Tesseract tự động
def configure_tesseract_path():
    """Tự động tìm và cấu hình đường dẫn Tesseract"""
    # Kiểm tra xem đã cấu hình chưa
    if hasattr(pytesseract.pytesseract, 'tesseract_cmd') and pytesseract.pytesseract.tesseract_cmd:
        return
    
    # Thử tìm trong PATH trước
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return
    
    # Nếu không tìm thấy trong PATH, thử các đường dẫn phổ biến
    if platform.system() == 'Darwin':  # macOS
        possible_paths = [
            '/opt/homebrew/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/usr/bin/tesseract'
        ]
    elif platform.system() == 'Windows':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        ]
    else:  # Linux
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract'
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

# Cấu hình khi import module
configure_tesseract_path()


def extract_text(image):
    """
    Trích xuất text từ ảnh bằng Tesseract OCR
    
    Args:
        image: numpy array (BGR image)
    
    Returns:
        str: Extracted text
    """
    all_texts = []
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Với ảnh rõ ràng, thử OCR trực tiếp trên ảnh gốc trước (không preprocessing)
    # Đôi khi preprocessing làm hỏng ảnh rõ
    
    preprocessing_variants = []
    
    # 1. Ảnh gốc grayscale (KHÔNG preprocessing) - tốt nhất cho ảnh rõ
    preprocessing_variants.append(("original_gray", gray))
    
    # 2. Resize lớn hơn để OCR đọc tốt hơn (nếu ảnh nhỏ)
    height, width = gray.shape
    if width < 800:  # Nếu ảnh nhỏ, resize lên
        scale = 800 / width
        large_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        preprocessing_variants.append(("large_gray", large_gray))
    
    # 3. Grayscale với contrast enhancement nhẹ
    enhanced_gray = enhance_contrast(gray, alpha=1.2)
    preprocessing_variants.append(("enhanced_gray", enhanced_gray))
    
    # 4. Denoise nhẹ (giữ text rõ)
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    preprocessing_variants.append(("denoised", denoised))
    
    # 5. Binary với OTSU (chỉ dùng khi cần)
    _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_variants.append(("otsu", otsu_binary))
    
    # PSM modes - ưu tiên mode tốt nhất cho thẻ sinh viên
    psm_modes = [
        ('--psm 6', 'block'),  # Uniform block of text - TỐT NHẤT cho thẻ
        ('--psm 4', 'column'),  # Single column
        ('--psm 3', 'auto'),  # Automatic page segmentation
        ('--psm 11', 'sparse'),  # Sparse text
    ]
    
    # Thử tất cả các combinations
    for variant_name, processed_img in preprocessing_variants:
        for psm_config, psm_name in psm_modes:
            try:
                text = pytesseract.image_to_string(
                    processed_img, 
                    lang='vie+eng', 
                    config=psm_config
                )
                if text.strip():
                    all_texts.append((text, variant_name, psm_name))
            except Exception as e:
                continue
    
    if not all_texts:
        # Fallback
        text = pytesseract.image_to_string(gray, lang='vie+eng', config='--psm 6')
        return text
    
    # Đánh giá chất lượng text
    scored_texts = []
    for text, variant, psm in all_texts:
        # Đếm số ký tự hợp lệ (chữ, số, dấu câu)
        valid_chars = len(re.findall(r'[A-Za-z0-9ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s\.,:;/\-]', text))
        # Đếm số (có thể là MSSV, ngày)
        numbers = len(re.findall(r'\d+', text))
        # Đếm từ có nghĩa (chữ tiếng Việt)
        words = len(re.findall(r'[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]{2,}', text))
        
        # Bonus cho variant "original_gray" (ưu tiên ảnh gốc)
        bonus = 100 if variant == "original_gray" else 0
        score = valid_chars * 1 + numbers * 5 + words * 10 + bonus
        scored_texts.append((score, text, variant, psm))
    
    # Sắp xếp theo score và lấy text tốt nhất
    scored_texts.sort(key=lambda x: x[0], reverse=True)
    
    if scored_texts:
        best_text = scored_texts[0][1]
        print(f"✓ Best OCR variant: {scored_texts[0][2]} with PSM {scored_texts[0][3]}, score: {scored_texts[0][0]}")
        return best_text
    
    return all_texts[0][0] if all_texts else ""


def parse_mssv(text):
    """
    Parse MSSV (Mã số sinh viên) từ text
    
    Args:
        text: str
    
    Returns:
        str: MSSV hoặc None
    """
    # Pattern cho MSSV (thường là số, có thể có chữ)
    # Ví dụ: 12345678, B1234567, 20220991, etc.
    patterns = [
        r'Mã SV:?\s*[^\d]*?([A-Z]?\d{6,10})',  # Format: Mã SV: 20220991 hoặc Ma SV ~20220991
        r'M[SS][SV]:?\s*[^\d]*?([A-Z]?\d{6,10})',
        r'Ma SV:?\s*[^\d]*?([A-Z]?\d{6,10})',  # Không dấu
        r'Mã số:?\s*([A-Z]?\d{6,10})',
        r'(?:STUDENT ID|MSSV|Mã SV):?\s*([A-Z]?\d{6,10})',
        r'\b([A-Z]?\d{8})\b',  # Generic pattern cho 8 số (format phổ biến)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            mssv = match.group(1).strip()
            # Validate: phải có ít nhất 6 chữ số
            if len(re.sub(r'[^0-9]', '', mssv)) >= 6:
                return mssv
    
    # Tìm tất cả các số trong text (kể cả số bị tách rời bởi ký tự đặc biệt)
    # Loại bỏ tất cả ký tự không phải số, chỉ giữ lại số
    numbers_only = re.findall(r'\d+', text)
    
    # Tìm số có 8 chữ số (MSSV) - ưu tiên số bắt đầu bằng 20xx
    for num in numbers_only:
        if len(num) == 8 and num.startswith('20'):  # MSSV thường bắt đầu bằng năm
            return num
    
    # Tìm các số gần nhau có thể ghép lại thành 8 chữ số
    for i in range(len(numbers_only) - 1):
        num1 = numbers_only[i]
        num2 = numbers_only[i + 1]
        # Kiểm tra xem có thể ghép không
        if len(num1) >= 4 and len(num2) >= 4:
            combined = num1 + num2
            if len(combined) == 8 and combined.startswith('20'):
                return combined
    
    # Tìm số có 6-10 chữ số và bắt đầu bằng 20
    for num in numbers_only:
        if 6 <= len(num) <= 10 and num.startswith('20'):
            if len(num) == 8:
                return num
            elif len(num) > 8:
                return num[:8]  # Lấy 8 chữ số đầu
            else:
                return num  # Nếu < 8 thì giữ nguyên
    
    # Fallback: Lấy số dài nhất có thể là MSSV (không nhất thiết bắt đầu bằng 20)
    long_numbers = [num for num in numbers_only if len(num) >= 6]
    if long_numbers:
        longest = max(long_numbers, key=len)
        if len(longest) >= 8:
            return longest[:8]
        return longest
    
    return None


def parse_ho_ten(text):
    """
    Parse Họ tên từ text
    
    Args:
        text: str
    
    Returns:
        str: Họ tên hoặc None
    """
    # Danh sách từ cần loại bỏ (tên trường đại học, etc.)
    exclude_words = [
        'UNIVERSITY', 'TECHNOLOGY', 'EAST', 'ASIA', 'OF', 'THE', 'AND',
        'TRƯỜNG', 'ĐẠI', 'HỌC', 'CÔNG', 'NGHỆ', 'ĐÔNG', 'Á',
        'UNIVERSITY OF TECHNOLOGY', 'EAST ASIA'
    ]
    
    # Pattern cho họ tên (thường có dấu tiếng Việt, có thể có dấu &)
    patterns = [
        # Pattern ưu tiên: "Ho & tê" hoặc "Họ & tên" (cho phép typo)
        r'[\\\/]?\s*Ho\s*[&]\s*t[êe]\.?\s*:?\s*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s]+?)(?:\n|Ngày|Mã|Niên|Date|ID|\d{1,2}[/-])',
        r'Họ\s*[&]\s*tên\.?\s*:?\s*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s\-]+?)(?:[:|]|\n|Ngày|Mã|Niên|Date|ID|\d{1,2}[/-])',
        r'Họ\s+[&]\s+tên\.?\s*:?\s*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s\-]+?)(?:[:|]|\n|Ngày|Mã|Niên|Date|ID|\d{1,2}[/-])',
        r'Họ\s+tên\.?\s*:?\s*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s\-]+?)(?:[:|]|\n|Ngày|Mã|Niên|Date|ID|\d{1,2}[/-])',
        r'Full Name:?\s*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s]+?)(?:\n|Date|ID|\d{1,2}[/-])',
        r'Tên:?\s*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s]+?)(?:\n|Ngày|Mã|\d{1,2}[/-])',
        # Pattern linh hoạt hơn - tìm "Họ" hoặc "tên" gần nhau
        r'(?:Họ|Tên|Ho)[\s&]*\.?\s*:?\s*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s\-]{3,30})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Loại bỏ các ký tự không hợp lệ ở đầu/cuối
            name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name)
            name = name.strip()
            # Loại bỏ dấu & nếu có
            name = re.sub(r'\s*&\s*', ' ', name)
            # Loại bỏ các ký tự đặc biệt và số, nhưng giữ dấu gạch ngang
            name = re.sub(r'[^\w\s\-]', '', name)
            # Thay dấu gạch ngang bằng khoảng trắng
            name = name.replace('-', ' ')
            # Loại bỏ các từ quá ngắn (< 2 ký tự) và chỉ giữ từ có chữ cái
            words = name.split()
            words = [w for w in words if len(w) >= 2 and re.search(r'[A-Za-zÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', w)]
            
            # Loại bỏ các từ trong exclude_words
            words = [w for w in words if w.upper() not in exclude_words]
            
            # Loại bỏ các từ ngắn không hợp lệ ở cuối tên (< 3 ký tự và không có dấu tiếng Việt)
            # Ví dụ: "Se", "N", "E" - những từ này thường là OCR noise
            if words:
                # Kiểm tra từ cuối cùng
                while len(words) > 0:
                    last_word = words[-1]
                    # Nếu từ cuối có < 3 ký tự và không chứa dấu tiếng Việt, loại bỏ
                    if len(last_word) < 3 and not re.search(r'[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', last_word):
                        words = words[:-1]
                    else:
                        break
            
            if words:
                name = ' '.join(words)
                # Kiểm tra xem có chứa từ loại trừ không
                name_upper = name.upper()
                if not any(exclude in name_upper for exclude in exclude_words):
                    if len(name) > 3:  # Tên phải có ít nhất 3 ký tự
                        return name
    
    # Fallback: Tìm chuỗi chữ hoa sau "Họ" hoặc "tên"
    # Tìm vị trí của "Họ" hoặc "tên"
    ho_ten_positions = []
    for match in re.finditer(r'Ho\s*[&]?\s*t[êe]|Họ\s*[&]?\s*tên', text, re.IGNORECASE):
        ho_ten_positions.append(match.end())
    
    # Tìm chuỗi chữ hoa ngay sau "Họ & tên"
    if ho_ten_positions:
        for pos in ho_ten_positions:
            # Lấy text sau vị trí "Họ & tên" - tăng lên 150 ký tự để lấy đủ tên dài
            remaining_text = text[pos:pos+150]  # Lấy 150 ký tự tiếp theo
            
            # Tìm chuỗi chữ hoa (2-5 từ) - cho phép tên dài hơn
            # Pattern linh hoạt hơn, cho phép tên bị cắt hoặc có ký tự lạ
            name_match = re.search(
                r'([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]+(?:\s+[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]+){1,4})', 
                remaining_text
            )
            if name_match:
                name = name_match.group(1).strip()
                # Loại bỏ ký tự đặc biệt
                name = re.sub(r'[^A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s]', ' ', name)
                name = ' '.join(name.split())
                words = name.split()
                # Loại bỏ các từ trong exclude_words
                words = [w for w in words if w.upper() not in exclude_words and len(w) >= 2]
                
                # Loại bỏ các từ ngắn không hợp lệ ở cuối
                if words:
                    while len(words) > 0:
                        last_word = words[-1]
                        if len(last_word) < 3 and not re.search(r'[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', last_word):
                            words = words[:-1]
                        else:
                            break
                
                if words and len(words) >= 2 and len(words) <= 5:  # Cho phép tên có 5 từ
                    return ' '.join(words)
    
    # Fallback cuối: Tìm tất cả chuỗi chữ hoa, nhưng loại bỏ các từ exclude
    all_uppercase_sequences = re.findall(r'[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ][A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s\W]*[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', text)
    
    # Làm sạch các chuỗi tìm được
    cleaned_sequences = []
    for seq in all_uppercase_sequences:
        cleaned = re.sub(r'[^A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s]', ' ', seq)
        cleaned = ' '.join(cleaned.split())
        words = cleaned.split()
        valid_words = [w for w in words if len(w) >= 2]
        
        # Loại bỏ các từ trong exclude_words
        valid_words = [w for w in valid_words if w.upper() not in exclude_words]
        
        # Loại bỏ các từ ngắn không hợp lệ ở cuối (< 3 ký tự, không có dấu tiếng Việt)
        if valid_words:
            while len(valid_words) > 0:
                last_word = valid_words[-1]
                if len(last_word) < 3 and not re.search(r'[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', last_word):
                    valid_words = valid_words[:-1]
                else:
                    break
        
        # Cho phép tên có 2-5 từ (có thể là họ + tên đệm + tên)
        if len(valid_words) >= 2 and len(valid_words) <= 5:
            cleaned_sequences.append(' '.join(valid_words))
    
    # Tìm chuỗi tốt nhất (2-5 từ, không chứa từ loại trừ)
    if cleaned_sequences:
        # Ưu tiên chuỗi có từ 2-5 từ, ưu tiên dài hơn một chút (có thể là tên đầy đủ)
        # Nhưng vẫn loại bỏ chuỗi quá dài (tên trường)
        best = None
        for seq in cleaned_sequences:
            word_count = len(seq.split())
            seq_upper = seq.upper()
            
            # Bỏ qua nếu chứa từ loại trừ
            if any(exclude in seq_upper for exclude in exclude_words):
                continue
            
            if 2 <= word_count <= 5:
                # Ưu tiên chuỗi có 3-4 từ (tên đầy đủ thường có 3-4 từ)
                # Nhưng không quá dài
                if len(seq) <= 35:  # Giới hạn độ dài
                    if best is None:
                        best = seq
                    elif word_count == 3 or word_count == 4:
                        # Ưu tiên 3-4 từ
                        if len(best.split()) < 3 or len(best.split()) > 4:
                            best = seq
                    elif word_count > len(best.split()):
                        # Nếu cả 2 đều không phải 3-4 từ, ưu tiên dài hơn
                        if len(seq) < len(best):
                            best = seq
        
        if best:
            return best
    
    return None


def parse_ngay_sinh(text):
    """
    Parse Ngày sinh từ text
    
    Args:
        text: str
    
    Returns:
        str: Ngày sinh (format: YYYY-MM-DD) hoặc None
    """
    # Pattern cho ngày sinh (dd/mm/yyyy hoặc dd-mm-yyyy)
    # Format trong thẻ: Ngày sinh: 02/05/2004
    patterns = [
        r'Ngày sinh\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # Cho phép khoảng trắng sau "Ngày sinh"
        r'Date of Birth:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'Sinh:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',  # Generic pattern
    ]
    
    dates_found = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Parse date
                if '/' in match:
                    day, month, year = match.split('/')
                elif '-' in match:
                    day, month, year = match.split('-')
                else:
                    continue
                
                day = int(day)
                month = int(month)
                year = int(year)
                
                # Validate date - ưu tiên năm từ 2000-2010 (sinh viên thường sinh trong khoảng này)
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2010:
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    # Verify date is valid
                    datetime.strptime(date_str, '%Y-%m-%d')
                    dates_found.append((year, date_str))
            except (ValueError, AttributeError):
                continue
    
    # Nếu không tìm thấy với pattern, tìm các số có thể ghép lại thành ngày
    # Tìm tất cả các số trong text
    all_numbers = re.findall(r'\d+', text)
    
    # Tìm pattern: số có 1-2 chữ số / số có 1-2 chữ số / số có 4 chữ số (năm)
    # Với khoảng cách ngắn giữa chúng
    for i in range(len(all_numbers) - 2):
        num1, num2, num3 = all_numbers[i], all_numbers[i+1], all_numbers[i+2]
        
        # Kiểm tra xem có thể là ngày sinh không (dd/mm/yyyy)
        if (len(num1) == 1 or len(num1) == 2) and \
           (len(num2) == 1 or len(num2) == 2) and \
           len(num3) == 4:
            try:
                day = int(num1)
                month = int(num2)
                year = int(num3)
                
                if 1 <= month <= 12 and 1 <= day <= 31 and 2000 <= year <= 2010:
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    datetime.strptime(date_str, '%Y-%m-%d')
                    dates_found.append((year, date_str))
            except:
                continue
    
    # Ưu tiên năm gần 2004 (nếu có nhiều ngày, chọn ngày phù hợp với sinh viên)
    if dates_found:
        # Sắp xếp theo năm gần 2004
        dates_found.sort(key=lambda x: abs(x[0] - 2004))
        return dates_found[0][1]
    
    return None


def parse_nien_khoa(text):
    """
    Parse Niên khóa từ text
    
    Args:
        text: str
    
    Returns:
        str: Niên khóa (format: YYYY-YYYY) hoặc None
    """
    # Pattern cho niên khóa: 2022-2027
    patterns = [
        r'Niên khóa\s*:?\s*(\d{4}[-/]\d{4})',  # Cho phép khoảng trắng sau "Niên khóa"
        r'Academic Year:?\s*(\d{4}[-/]\d{4})',
        r'\b(\d{4}[-/]\d{4})\b',  # Generic pattern
    ]
    
    nien_khoas_found = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            nien_khoa = match.strip()
            # Normalize format: YYYY-YYYY
            nien_khoa = nien_khoa.replace('/', '-')
            # Validate: năm sau phải lớn hơn năm trước
            parts = nien_khoa.split('-')
            if len(parts) == 2:
                try:
                    year1 = int(parts[0])
                    year2 = int(parts[1])
                    # Kiểm tra khoảng cách hợp lý (thường 4-6 năm)
                    if 2000 <= year1 <= 2100 and year1 < year2 and (year2 - year1) <= 6:
                        nien_khoas_found.append((year1, nien_khoa))
                except ValueError:
                    continue
    
    if nien_khoas_found:
        # Ưu tiên niên khóa gần 2022
        nien_khoas_found.sort(key=lambda x: abs(x[0] - 2022))
        return nien_khoas_found[0][1]
    
    # Nếu không tìm thấy với pattern, tìm 2 số có 4 chữ số gần nhau
    # Tìm tất cả số có 4 chữ số và vị trí của chúng
    all_year_matches = list(re.finditer(r'\d{4}', text))
    all_numbers = [m.group() for m in all_year_matches]
    all_positions = [m.start() for m in all_year_matches]
    
    for i in range(len(all_numbers) - 1):
        year1_str = all_numbers[i]
        year2_str = all_numbers[i + 1]
        pos1 = all_positions[i]
        pos2 = all_positions[i + 1]
        
        # Kiểm tra khoảng cách giữa 2 năm (phải gần nhau, không quá 20 ký tự)
        distance = pos2 - pos1 - len(year1_str)
        
        try:
            year1 = int(year1_str)
            year2 = int(year2_str)
            
            # Kiểm tra xem có phải niên khóa không
            # Cho phép khoảng cách lớn hơn (có thể có ký tự OCR noise giữa)
            if (2000 <= year1 <= 2100 and year1 < year2 and 
                (year2 - year1) <= 6 and distance <= 20):
                nien_khoa = f"{year1}-{year2}"
                nien_khoas_found.append((year1, nien_khoa))
        except:
            continue
    
    if nien_khoas_found:
        nien_khoas_found.sort(key=lambda x: abs(x[0] - 2022))
        return nien_khoas_found[0][1]
    
    # Fallback cuối: Tìm cặp năm gần nhau nhất có thể là niên khóa
    # Không cần từ khóa "Niên khóa", chỉ cần tìm pattern năm-năm
    if len(all_numbers) >= 2:
        candidates = []
        for i in range(len(all_numbers) - 1):
            year1_str = all_numbers[i]
            year2_str = all_numbers[i + 1]
            
            try:
                year1 = int(year1_str)
                year2 = int(year2_str)
                
                # Tìm các cặp năm hợp lý cho niên khóa
                if (2000 <= year1 <= 2030 and year1 < year2 and 
                    (year2 - year1) >= 3 and (year2 - year1) <= 7):
                    # Kiểm tra xem có phải năm gần hiện tại không (2020-2030)
                    if 2020 <= year1 <= 2030:
                        nien_khoa = f"{year1}-{year2}"
                        candidates.append((year1, nien_khoa))
            except:
                continue
        
        if candidates:
            # Ưu tiên năm gần 2022
            candidates.sort(key=lambda x: abs(x[0] - 2022))
            return candidates[0][1]
    
    return None


def parse_ngay_het_han(text):
    """
    Parse Thẻ có giá trị đến ngày từ text
    
    Args:
        text: str
    
    Returns:
        str: Ngày hết hạn (format: YYYY-MM-DD) hoặc None
    """
    # Pattern cho thẻ có giá trị đến ngày: 31/12/2027
    patterns = [
        r':?\s*Th[ẻe]\s+có\s+giá\s+tr[ịrđê]+[^\d]*?ngày:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # ": The có giá trrđên ngày:"
        r'Thẻ\s+có\s+giá\s+tr[ịrđê]+[^\d]*?ngày:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # Cho phép typo "trrđên", "trị đến", etc.
        r':?\s*Th[ẻe]\s+có\s+giá\s+tr[ịrđê]+[^\d]*?:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # Không cần "ngày"
        r'Card valid until:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'GOOD THRU:?\s*(\d{1,2}[-/]\d{2,4})',  # Format: 11/28
        r'VALID TO:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'VALID FROM:?\s*\d{1,2}[-/]\d{2,4}.*?GOOD THRU:?\s*(\d{1,2}[-/]\d{2,4})',  # Tìm sau VALID FROM
    ]
    
    dates_found = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Parse date
                if '/' in match:
                    parts = match.split('/')
                elif '-' in match:
                    parts = match.split('-')
                else:
                    continue
                
                # Handle GOOD THRU format (11/28) - chỉ có tháng/năm
                if len(parts) == 2:
                    month, year = parts
                    # Default to last day of month
                    day = 31  # Convert to int immediately
                    month = int(month)
                    year = int(year)
                    # Handle 2-digit year
                    if year < 100:
                        if year > 50:  # Nếu > 50 thì là 19xx
                            year = 1900 + year
                        else:  # Nếu <= 50 thì là 20xx
                            year = 2000 + year
                elif len(parts) == 3:
                    day, month, year = parts
                    day = int(day)
                    month = int(month)
                    year = int(year)
                else:
                    continue
                
                # Validate date - ưu tiên năm trong tương lai (2027, 2028)
                if 1 <= month <= 12 and 1 <= day <= 31 and 2020 <= year <= 2100:
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    # Verify date is valid
                    datetime.strptime(date_str, '%Y-%m-%d')
                    dates_found.append((year, date_str))
            except (ValueError, AttributeError):
                continue
    
    if dates_found:
        # Ưu tiên năm trong tương lai (thẻ hết hạn)
        dates_found.sort(key=lambda x: x[0], reverse=True)
        return dates_found[0][1]
    
    # Fallback: Tìm tất cả các ngày có năm > 2020 (có thể là ngày hết hạn)
    # Tránh lấy ngày sinh (thường năm 2000-2010)
    all_date_patterns = [
        r'\b(\d{1,2}[-/]\d{1,2}[-/]20[2-9]\d)\b',  # Năm từ 2020-2099
        r'(\d{1,2}[-/]\d{1,2}[-/]20[3-9]\d)',  # Năm từ 2023-2099 (không cần word boundary)
    ]
    
    fallback_dates = []
    
    for date_pattern in all_date_patterns:
        matches = re.findall(date_pattern, text)
        for match in matches:
            try:
                if '/' in match:
                    parts = match.split('/')
                elif '-' in match:
                    parts = match.split('-')
                else:
                    continue
                
                if len(parts) == 3:
                    day, month, year = parts
                    day = int(day)
                    month = int(month)
                    year = int(year)
                    
                    # Validate date - ưu tiên năm > 2020 (thẻ hết hạn)
                    if 1 <= month <= 12 and 1 <= day <= 31 and 2020 <= year <= 2100:
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        datetime.strptime(date_str, '%Y-%m-%d')
                        fallback_dates.append((year, date_str))
            except (ValueError, AttributeError):
                continue
    
    if fallback_dates:
        # Ưu tiên năm xa nhất (thẻ hết hạn)
        fallback_dates.sort(key=lambda x: x[0], reverse=True)
        return fallback_dates[0][1]
    
    return None


def extract_student_info(image):
    """
    Trích xuất tất cả thông tin sinh viên từ ảnh thẻ
    
    Args:
        image: numpy array (BGR image)
    
    Returns:
        dict: Dictionary chứa các trường:
            - mssv: str
            - ho_ten: str
            - ngay_sinh: str (YYYY-MM-DD)
            - nien_khoa: str (YYYY-YYYY)
            - ngay_het_han: str (YYYY-MM-DD)
            - raw_text: str (text gốc từ OCR)
    """
    # Extract raw text
    raw_text = extract_text(image)
    
    # Debug: in raw text để kiểm tra
    print(f"\n=== Raw OCR Text (first 500 chars) ===")
    print(raw_text[:500])
    print("=======================================\n")
    
    # Parse các trường
    info = {
        'mssv': parse_mssv(raw_text),
        'ho_ten': parse_ho_ten(raw_text),
        'ngay_sinh': parse_ngay_sinh(raw_text),
        'nien_khoa': parse_nien_khoa(raw_text),
        'ngay_het_han': parse_ngay_het_han(raw_text),
        'raw_text': raw_text
    }
    
    # Debug: in kết quả parse
    print("=== Parsed Results ===")
    for key, value in info.items():
        if key != 'raw_text':
            print(f"{key}: {value}")
    print("======================\n")
    
    return info

