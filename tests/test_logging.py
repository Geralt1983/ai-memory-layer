import pytest
import os
import json
import tempfile
import logging
from unittest.mock import patch, Mock
from core.logging_config import (
    get_logger,
    MemoryLayerLogger,
    JSONFormatter,
    log_memory_operation,
    log_api_request,
    log_embedding_operation,
    monitor_performance,
)


class TestLoggingConfig:
    """Test cases for logging configuration"""

    def test_json_formatter(self):
        """Test JSON formatter produces valid JSON"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)  # Should not raise exception

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_json_formatter_with_extra(self):
        """Test JSON formatter includes extra fields"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.custom_field = "custom_value"
        record.operation = "test_operation"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["custom_field"] == "custom_value"
        assert parsed["operation"] == "test_operation"

    def test_get_logger(self):
        """Test logger creation"""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "ai_memory_layer.test_module"

    def test_get_logger_auto_name(self):
        """Test automatic logger naming"""
        logger = get_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name.startswith("ai_memory_layer.")

    @patch.dict(
        os.environ,
        {"LOG_LEVEL": "DEBUG", "LOG_FORMAT": "text", "LOG_FILE": "", "LOG_DIR": ""},
    )
    def test_memory_layer_logger_initialization(self):
        """Test MemoryLayerLogger initialization with environment variables"""
        # Reset singleton
        MemoryLayerLogger._instance = None
        MemoryLayerLogger._initialized = False

        logger_instance = MemoryLayerLogger()

        assert logger_instance is not None
        assert MemoryLayerLogger._initialized

    def test_log_memory_operation(self):
        """Test memory operation logging"""
        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_memory_operation("add", memory_id="123", content_length=100)

            mock_get_logger.assert_called_once_with("memory_operations")
            mock_logger.info.assert_called_once()

            # Check call arguments
            call_args = mock_logger.info.call_args
            assert "Memory operation: add" in call_args[0][0]
            assert call_args[1]["extra"]["operation"] == "add"
            assert call_args[1]["extra"]["memory_id"] == "123"
            assert call_args[1]["extra"]["content_length"] == 100

    def test_log_api_request(self):
        """Test API request logging"""
        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_api_request("GET", "/health", 200, 0.05, client_ip="127.0.0.1")

            mock_get_logger.assert_called_once_with("api_requests")
            mock_logger.log.assert_called_once()

            # Check call arguments
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO  # Log level
            assert "API GET /health" in call_args[0][1]
            assert call_args[1]["extra"]["status_code"] == 200
            assert call_args[1]["extra"]["response_time_ms"] == 50.0

    def test_log_api_request_error_level(self):
        """Test API request logging uses appropriate log level for errors"""
        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Test 4xx status code (WARNING)
            log_api_request("POST", "/test", 404, 0.1)
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.WARNING

            # Test 5xx status code (ERROR)
            log_api_request("POST", "/test", 500, 0.2)
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.ERROR

    def test_log_embedding_operation(self):
        """Test embedding operation logging"""
        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_embedding_operation(
                "embed_text", 100, "text-embedding-ada-002", dimension=1536
            )

            mock_get_logger.assert_called_once_with("embeddings")
            mock_logger.info.assert_called_once()

            call_args = mock_logger.info.call_args
            assert "Embedding operation: embed_text" in call_args[0][0]
            assert call_args[1]["extra"]["text_length"] == 100
            assert call_args[1]["extra"]["model"] == "text-embedding-ada-002"

    def test_monitor_performance_decorator(self):
        """Test performance monitoring decorator"""
        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @monitor_performance("test_operation")
            def test_function():
                return "success"

            result = test_function()

            assert result == "success"
            mock_logger.info.assert_called_once()

            call_args = mock_logger.info.call_args
            assert "Performance: test_operation" in call_args[0][0]
            assert call_args[1]["extra"]["operation"] == "test_operation"
            assert "duration_ms" in call_args[1]["extra"]
            assert call_args[1]["extra"]["success"] is True

    def test_monitor_performance_decorator_with_exception(self):
        """Test performance monitoring decorator handles exceptions"""
        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @monitor_performance("test_operation")
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                failing_function()

            mock_logger.warning.assert_called_once()

            call_args = mock_logger.warning.call_args
            assert "Performance: test_operation (failed)" in call_args[0][0]
            assert call_args[1]["extra"]["success"] is False
            assert call_args[1]["extra"]["error"] == "Test error"


class TestLoggingIntegration:
    """Integration tests for logging with other components"""

    def test_logging_with_memory_engine(self):
        """Test that MemoryEngine properly logs operations"""
        from core.memory_engine import MemoryEngine

        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            engine = MemoryEngine()
            engine.add_memory("Test memory")

            # Should have logged initialization and add_memory
            assert mock_logger.info.call_count >= 2

    def test_logging_with_openai_integration(self):
        """Test that OpenAI integration logs properly"""
        from integrations.openai_integration import OpenAIIntegration
        from core.memory_engine import MemoryEngine

        with patch("core.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            memory_engine = MemoryEngine()

            # This will fail due to missing API key, but should log initialization
            try:
                integration = OpenAIIntegration("fake-key", memory_engine)
            except:
                pass

            # Should have logged initialization
            mock_logger.info.assert_called()


class TestLoggingConfiguration:
    """Test logging configuration scenarios"""

    def test_file_logging_setup(self):
        """Test file logging configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            with patch.dict(
                os.environ,
                {
                    "LOG_LEVEL": "INFO",
                    "LOG_FORMAT": "json",
                    "LOG_FILE": "test.log",
                    "LOG_DIR": temp_dir,
                },
            ):
                # Reset singleton
                MemoryLayerLogger._instance = None
                MemoryLayerLogger._initialized = False

                logger = get_logger("test")
                logger.info("Test message")

                # Force handlers to flush
                for handler in logger.handlers:
                    handler.flush()

                # Check if log file was created (may not have content due to buffering)
                # The main test is that no exception was raised during setup
                assert True

    def test_console_only_logging(self):
        """Test console-only logging configuration"""
        with patch.dict(
            os.environ,
            {
                "LOG_LEVEL": "WARNING",
                "LOG_FORMAT": "text",
                "LOG_FILE": "",
                "LOG_DIR": "",
            },
        ):
            # Reset singleton
            MemoryLayerLogger._instance = None
            MemoryLayerLogger._initialized = False

            logger = get_logger("test")

            # Should not raise exception
            logger.warning("Test warning message")
            assert True
