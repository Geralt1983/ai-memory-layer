"""
Structured logging configuration for AI Memory Layer
"""
import logging
import logging.config
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add process and thread info
        log_entry["process_id"] = os.getpid()
        log_entry["thread_id"] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'lineno', 'funcName', 'created', 
                'msecs', 'relativeCreated', 'thread', 'threadName', 
                'processName', 'process', 'exc_info', 'exc_text', 'stack_info'
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class MemoryLayerLogger:
    """Centralized logger for the AI Memory Layer"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            self._initialized = True
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Get configuration from environment
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_format = os.getenv("LOG_FORMAT", "json").lower()  # json or text
        log_file = os.getenv("LOG_FILE", None)
        log_dir = os.getenv("LOG_DIR", "./logs")
        
        # Create logs directory if it doesn't exist
        if log_file or log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure formatters
        formatters = {
            "json": {
                "()": JSONFormatter,
            },
            "text": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        }
        
        # Configure handlers
        handlers = {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": log_format,
                "stream": "ext://sys.stdout"
            }
        }
        
        # Add file handler if specified
        if log_file:
            handlers["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": log_format,
                "filename": os.path.join(log_dir, log_file),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            }
        
        # Configure loggers
        loggers = {
            "ai_memory_layer": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "propagate": False
            },
            "api": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            }
        }
        
        # Root logger configuration
        root_config = {
            "level": log_level,
            "handlers": list(handlers.keys())
        }
        
        # Complete logging configuration
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "loggers": loggers,
            "root": root_config
        }
        
        # Apply configuration
        logging.config.dictConfig(config)
        
        # Log startup message
        logger = logging.getLogger("ai_memory_layer.startup")
        logger.info("Logging system initialized", extra={
            "log_level": log_level,
            "log_format": log_format,
            "log_file": log_file,
            "handlers": list(handlers.keys())
        })


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance with consistent configuration"""
    # Ensure logging is initialized
    MemoryLayerLogger()
    
    if name is None:
        # Use the calling module name
        frame = sys._getframe(1)
        module = frame.f_globals.get('__name__', 'unknown')
        name = f"ai_memory_layer.{module.split('.')[-1]}"
    elif not name.startswith('ai_memory_layer'):
        name = f"ai_memory_layer.{name}"
    
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs):
    """Decorator factory for logging function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            # Log function entry
            logger.debug(f"Entering {func_name}", extra={
                "function": func_name,
                "args_count": len(args),
                "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}  # Truncate long values
            })
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name}", extra={
                    "function": func_name,
                    "success": True
                })
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}", extra={
                    "function": func_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_memory_operation(operation: str, memory_id: Optional[str] = None, **extra):
    """Log memory-related operations with consistent structure"""
    logger = get_logger("memory_operations")
    logger.info(f"Memory operation: {operation}", extra={
        "operation": operation,
        "memory_id": memory_id,
        **extra
    })


def log_api_request(method: str, path: str, status_code: int, response_time: float, **extra):
    """Log API requests with consistent structure"""
    logger = get_logger("api_requests")
    
    level = logging.INFO
    if status_code >= 400:
        level = logging.WARNING
    if status_code >= 500:
        level = logging.ERROR
    
    logger.log(level, f"API {method} {path}", extra={
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_time_ms": round(response_time * 1000, 2),
        **extra
    })


def log_embedding_operation(operation: str, text_length: int, model: str, **extra):
    """Log embedding operations"""
    logger = get_logger("embeddings")
    logger.info(f"Embedding operation: {operation}", extra={
        "operation": operation,
        "text_length": text_length,
        "model": model,
        **extra
    })


def log_vector_store_operation(operation: str, store_type: str, **extra):
    """Log vector store operations"""
    logger = get_logger("vector_store")
    logger.info(f"Vector store operation: {operation}", extra={
        "operation": operation,
        "store_type": store_type,
        **extra
    })


# Performance monitoring decorator
def monitor_performance(operation: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            logger = get_logger("performance")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"Performance: {operation}", extra={
                    "operation": operation,
                    "duration_ms": round(duration * 1000, 2),
                    "success": True
                })
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.warning(f"Performance: {operation} (failed)", extra={
                    "operation": operation,
                    "duration_ms": round(duration * 1000, 2),
                    "success": False,
                    "error": str(e)
                })
                raise
        
        return wrapper
    return decorator