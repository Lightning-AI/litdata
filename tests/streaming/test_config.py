import pytest
from litdata.streaming.config import load_subsampled_chunks


def test_load_subsampled_chunks():
    my_subsampled_files = ["1.txt", "2.txt", "5.txt", "3.txt", "9.txt"]

    original_chunks = [
        {"foo": "a", "filename": "1.txt"},
        {"foo": "b", "filename": "2.txt"},
        {"foo": "c", "filename": "3.txt"},
        {"foo": "d", "filename": "4.txt"},
        {"foo": "e", "filename": "5.txt"},
        {"foo": "f", "filename": "6.txt"},
        {"foo": "g", "filename": "7.txt"},
        {"foo": "h", "filename": "8.txt"},
        {"foo": "i", "filename": "9.txt"},
    ]

    assert load_subsampled_chunks(my_subsampled_files, original_chunks) == [
        {"foo": "a", "filename": "1.txt"},
        {"foo": "b", "filename": "2.txt"},
        {"foo": "e", "filename": "5.txt"},
        {"foo": "c", "filename": "3.txt"},
        {"foo": "i", "filename": "9.txt"},
    ]

    my_subsampled_files = ["1.txt", "21.txt", "5.txt", "3.txt", "9.txt"]

    with pytest.raises(ValueError, match="Mismatch in subsampled files"):
        load_subsampled_chunks(my_subsampled_files, original_chunks)
