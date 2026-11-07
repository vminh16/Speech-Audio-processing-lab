"""
Text Normalization Module for Domain-Specific Queuing TTS.
CSE457 - Lab 04: Text-to-Speech
"""

import re
from typing import List, Set

# Bảng ánh xạ từng chữ số sang từ tiếng Việt trong miền thông báo xếp hàng
DIGIT_TO_WORD_MAP = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

# Tập từ vựng cốt lõi của miền hẹp "thông báo xếp hàng"
QUEUE_VOCAB: Set[str] = {
    "mời",
    "khách",
    "hàng",
    "số",
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
    "đến",
    "quầy",
}


def digit_to_words(digit_str: str) -> List[str]:
    """
    Ánh xạ chuỗi chữ số thành danh sách từ phát âm từng chữ số.
    Ví dụ: '23' -> ['hai', 'ba'], '08' -> ['không', 'tám'].
    """
    words = []
    for char in digit_str:
        if char in DIGIT_TO_WORD_MAP:
            words.append(DIGIT_TO_WORD_MAP[char])
    return words


def is_valid_queue_token(token: str) -> bool:
    """Kiểm tra token có thuộc tập từ vựng hợp lệ của miền xếp hàng hay không."""
    return token.lower() in QUEUE_VOCAB


def normalize_queue_text(text: str) -> List[str]:
    """
    Chuẩn hóa văn bản thông báo xếp hàng và ánh xạ sang chuỗi token phát âm.

    Quy trình:
    1. Chuyển thành chữ thường (lowercase).
    2. Loại bỏ dấu câu và ký tự đặc biệt (.,;:!?-#...).
    3. Tách theo khoảng trắng.
    4. Phân tích từng token:
       - Nếu là chuỗi chữ số (hoặc chứa chữ số): tách thành các chữ số đơn lẻ và ánh xạ sang từ.
       - Nếu là từ chữ: giữ nguyên nếu hợp lệ.

    Ví dụ:
        'Mời khách hàng số 23 đến quầy 5.'
        -> ['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']
    """
    if not text:
        return []

    # 1. Chuyển chữ thường
    text_clean = text.lower()

    # 2. Thay thế dấu câu và ký tự không phải chữ/số tiếng Việt bằng khoảng trắng
    # Giữ lại các ký tự tiếng Việt có dấu, chữ cái latin thường và chữ số 0-9
    text_clean = re.sub(
        r"[^a-z0-9àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]",
        " ",
        text_clean,
    )

    # 3. Tách từ theo khoảng trắng
    raw_tokens = text_clean.split()
    normalized_tokens: List[str] = []

    for tok in raw_tokens:
        # Nếu token là chuỗi chữ số (ví dụ: '23', '08', '5')
        if tok.isdigit():
            normalized_tokens.extend(digit_to_words(tok))
        # Nếu token chứa cả chữ và số (ví dụ: 'số23')
        elif re.search(r"\d", tok):
            sub_parts = re.findall(r"\d+|[a-zà-ỹđ]+", tok)
            for part in sub_parts:
                if part.isdigit():
                    normalized_tokens.extend(digit_to_words(part))
                else:
                    normalized_tokens.append(part)
        else:
            normalized_tokens.append(tok)

    return normalized_tokens
