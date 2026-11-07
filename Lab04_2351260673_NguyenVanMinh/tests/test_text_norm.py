"""
Unit tests for text normalization module in Lab 04 TTS.
Following Test-Driven Development (TDD).
"""

import pytest
from src.text_norm import (
    normalize_queue_text,
    digit_to_words,
    QUEUE_VOCAB,
    is_valid_queue_token,
)


def test_standard_queue_sentence():
    """Kiểm tra câu thông báo chuẩn theo SPEC.md."""
    text = "Mời khách hàng số 23 đến quầy 5."
    expected = ["mời", "khách", "hàng", "số", "hai", "ba", "đến", "quầy", "năm"]
    assert normalize_queue_text(text) == expected


def test_uppercase_and_punctuation():
    """Kiểm tra xử lý chữ hoa, dấu câu chấm than, phẩy, và số có chữ số 0 đứng trước."""
    text = "MỜI KHÁCH HÀNG, SỐ 08 ĐẾN QUẦY 10!"
    expected = [
        "mời",
        "khách",
        "hàng",
        "số",
        "không",
        "tám",
        "đến",
        "quầy",
        "một",
        "không",
    ]
    assert normalize_queue_text(text) == expected


def test_spacing_and_special_symbols():
    """Kiểm tra khoảng trắng thừa và ký tự đặc biệt như #, -, :"""
    text = "  Mời   khách hàng   số: #42 - đến  quầy 3... "
    expected = ["mời", "khách", "hàng", "số", "bốn", "hai", "đến", "quầy", "ba"]
    assert normalize_queue_text(text) == expected


def test_all_digits_mapping():
    """Kiểm tra ánh xạ toàn bộ các chữ số từ 0 đến 9."""
    for d, word in [
        ("0", "không"),
        ("1", "một"),
        ("2", "hai"),
        ("3", "ba"),
        ("4", "bốn"),
        ("5", "năm"),
        ("6", "sáu"),
        ("7", "bảy"),
        ("8", "tám"),
        ("9", "chín"),
    ]:
        assert digit_to_words(d) == [word]

    assert digit_to_words("0123456789") == [
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
    ]


def test_queue_vocab_completeness():
    """Kiểm tra tập từ vựng miền hẹp đủ 16 từ cốt lõi."""
    expected_words = {
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
    assert expected_words.issubset(QUEUE_VOCAB)
    for word in expected_words:
        assert is_valid_queue_token(word) is True
