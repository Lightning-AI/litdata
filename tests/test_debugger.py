import logging

import pytest


def test_get_logger_level():
    from litdata.debugger import get_logger_level

    assert get_logger_level("DEBUG") == logging.DEBUG
    assert get_logger_level("INFO") == logging.INFO
    assert get_logger_level("WARNING") == logging.WARNING
    assert get_logger_level("ERROR") == logging.ERROR
    assert get_logger_level("CRITICAL") == logging.CRITICAL
    with pytest.raises(ValueError, match="Invalid log level"):
        get_logger_level("INVALID")
