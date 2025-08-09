"""
Structured logging configuration for AI Memory Layer
"""

import logging
import logging.config
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional, Callable
from pathlib import Path

# Try to import prometheus_client for metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock metrics for compatibility
    class MockMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    Counter = Histogram = Gauge = Info = lambda *args, **kwargs: MockMetric()


# Prometheus metrics definitions
METRICS = {}

def init_prometheus_metrics():
    """Initialize Prometheus metrics"""
    global METRICS
    
    METRICS = {
        # Memory operations
        'memory_operations_total': Counter(
            'ai_memory_layer_memory_operations_total',
            'Total number of memory operations',
            ['operation', 'status']
        ),
        'memory_operation_duration_seconds': Histogram(
            'ai_memory_layer_memory_operation_duration_seconds',
            'Duration of memory operations',
            ['operation']
        ),
        'memories_total': Gauge(
            'ai_memory_layer_memories_total',
            'Total number of memories stored'
        ),
        
        # Embedding operations  
        'embedding_operations_total': Counter(
            'ai_memory_layer_embedding_operations_total',
            'Total number of embedding operations',
            ['provider', 'model', 'status']
        ),
        'embedding_operation_duration_seconds': Histogram(
            'ai_memory_layer_embedding_operation_duration_seconds',
            'Duration of embedding operations',
            ['provider', 'model']
        ),
        'embedding_tokens_processed': Counter(
            'ai_memory_layer_embedding_tokens_processed_total',
            'Total number of tokens processed for embeddings',
            ['provider', 'model']
        ),
        
        # API operations
        'api_requests_total': Counter(
            'ai_memory_layer_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code']
        ),
        'api_request_duration_seconds': Histogram(
            'ai_memory_layer_api_request_duration_seconds',
            'Duration of API requests',
            ['method', 'endpoint']
        ),
        
        # Vector store operations
        'vector_store_operations_total': Counter(
            'ai_memory_layer_vector_store_operations_total',
            'Total number of vector store operations',
            ['operation', 'store_type', 'status']
        ),
        'vector_store_size': Gauge(
            'ai_memory_layer_vector_store_size',
            'Number of vectors in the store',
            ['store_type']
        ),
        
        # Cache operations
        'cache_operations_total': Counter(
            'ai_memory_layer_cache_operations_total',
            'Total number of cache operations',
            ['operation', 'cache_type', 'result']
        ),
        'cache_hit_ratio': Gauge(
            'ai_memory_layer_cache_hit_ratio',
            'Cache hit ratio',
            ['cache_type']
        ),
        
        # System info
        'system_info': Info(
            'ai_memory_layer_system_info',
            'System information'
        ),
        'build_info': Info(
            'ai_memory_layer_build_info', 
            'Build information'
        )
    }

# Initialize metrics on module load
init_prometheus_metrics()


class PrometheusJSONFormatter(logging.Formatter):
    """JSON formatter that also updates Prometheus metrics"""

    def format(self, record: logging.LogRecord) -> str:
        # Update Prometheus metrics based on log record
        self._update_metrics_from_log(record)
        
        # Use standard JSON formatting
        return JSONFormatter().format(record)
    
    def _update_metrics_from_log(self, record: logging.LogRecord):
        """Update Prometheus metrics based on log records"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Extract operation info from extra fields
            operation = getattr(record, 'operation', None)
            
            # Memory operations
            if hasattr(record, 'memory_id') or 'memory' in record.name.lower():
                status = 'success' if record.levelno < logging.WARNING else 'error'
                METRICS['memory_operations_total'].labels(
                    operation=operation or 'unknown',
                    status=status
                ).inc()
                
                if hasattr(record, 'duration_ms'):
                    METRICS['memory_operation_duration_seconds'].labels(
                        operation=operation or 'unknown'
                    ).observe(record.duration_ms / 1000.0)
            
            # API requests
            if hasattr(record, 'method') and hasattr(record, 'path'):
                status_code = str(getattr(record, 'status_code', 'unknown'))
                METRICS['api_requests_total'].labels(
                    method=record.method,
                    endpoint=record.path,
                    status_code=status_code
                ).inc()
                
                if hasattr(record, 'response_time_ms'):
                    METRICS['api_request_duration_seconds'].labels(
                        method=record.method,
                        endpoint=record.path
                    ).observe(record.response_time_ms / 1000.0)
            
            # Embedding operations
            if 'embed' in record.name.lower() or hasattr(record, 'model'):
                provider = getattr(record, 'provider', 'unknown')
                model = getattr(record, 'model', 'unknown')
                status = 'success' if record.levelno < logging.WARNING else 'error'
                
                METRICS['embedding_operations_total'].labels(
                    provider=provider,
                    model=model,
                    status=status
                ).inc()
                
                if hasattr(record, 'duration_ms'):
                    METRICS['embedding_operation_duration_seconds'].labels(
                        provider=provider,
                        model=model
                    ).observe(record.duration_ms / 1000.0)
                
                if hasattr(record, 'text_length'):
                    # Rough token estimation (1 token ≈ 4 chars)
                    estimated_tokens = record.text_length // 4
                    METRICS['embedding_tokens_processed'].labels(
                        provider=provider,
                        model=model
                    ).inc(estimated_tokens)
        
        except Exception:
            # Don't let metric updates break logging
            pass


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
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
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class MemoryLayerLogger:
    """Centralized logger for the AI Memory Layer"""

    _instance = None
    _initialized = False

    def __new__(cls) -> "MemoryLayerLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self.setup_logging()
            self._initialized = True

    def setup_logging(self) -> None:
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
                "()": PrometheusJSONFormatter if PROMETHEUS_AVAILABLE else JSONFormatter,
            },
            "json_simple": {
                "()": JSONFormatter,
            },
            "text": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        }

        # Configure handlers
        handlers = {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": log_format,
                "stream": "ext://sys.stdout",
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
                "encoding": "utf8",
            }

        # Configure loggers
        loggers = {
            "ai_memory_layer": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "api": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
        }

        # Root logger configuration
        root_config = {"level": log_level, "handlers": list(handlers.keys())}

        # Complete logging configuration
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "loggers": loggers,
            "root": root_config,
        }

        # Apply configuration
        logging.config.dictConfig(config)

        # Log startup message
        logger = logging.getLogger("ai_memory_layer.startup")
        logger.info(
            "Logging system initialized",
            extra={
                "log_level": log_level,
                "log_format": log_format,
                "log_file": log_file,
                "handlers": list(handlers.keys()),
            },
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with consistent configuration"""
    # Ensure logging is initialized
    MemoryLayerLogger()

    if name is None:
        # Use the calling module name
        frame = sys._getframe(1)
        module = frame.f_globals.get("__name__", "unknown")
        name = f"ai_memory_layer.{module.split('.')[-1]}"
    elif not name.startswith("ai_memory_layer"):
        name = f"ai_memory_layer.{name}"

    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs) -> Callable:
    """Decorator factory for logging function calls"""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger()

            # Log function entry
            logger.debug(
                f"Entering {func_name}",
                extra={
                    "function": func_name,
                    "args_count": len(args),
                    "kwargs": {
                        k: str(v)[:100] for k, v in kwargs.items()
                    },  # Truncate long values
                },
            )

            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"Exiting {func_name}",
                    extra={"function": func_name, "success": True},
                )
                return result
            except Exception as e:
                logger.error(
                    f"Error in {func_name}: {str(e)}",
                    extra={
                        "function": func_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_memory_operation(
    operation: str, memory_id: Optional[str] = None, **extra: Any
) -> None:
    """Log memory-related operations with consistent structure"""
    logger = get_logger("memory_operations")
    logger.info(
        f"Memory operation: {operation}",
        extra={"operation": operation, "memory_id": memory_id, **extra},
    )


def log_api_request(
    method: str, path: str, status_code: int, response_time: float, **extra: Any
) -> None:
    """Log API requests with consistent structure"""
    logger = get_logger("api_requests")

    level = logging.INFO
    if status_code >= 400:
        level = logging.WARNING
    if status_code >= 500:
        level = logging.ERROR

    logger.log(
        level,
        f"API {method} {path}",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2),
            **extra,
        },
    )


def log_embedding_operation(
    operation: str, text_length: int, model: str, **extra: Any
) -> None:
    """Log embedding operations"""
    logger = get_logger("embeddings")
    logger.info(
        f"Embedding operation: {operation}",
        extra={
            "operation": operation,
            "text_length": text_length,
            "model": model,
            **extra,
        },
    )


def log_vector_store_operation(operation: str, store_type: str, **extra):
    """Log vector store operations"""
    logger = get_logger("vector_store")
    logger.info(
        f"Vector store operation: {operation}",
        extra={"operation": operation, "store_type": store_type, **extra},
    )


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

                logger.info(
                    f"Performance: {operation}",
                    extra={
                        "operation": operation,
                        "duration_ms": round(duration * 1000, 2),
                        "success": True,
                    },
                )

                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.warning(
                    f"Performance: {operation} (failed)",
                    extra={
                        "operation": operation,
                        "duration_ms": round(duration * 1000, 2),
                        "success": False,
                        "error": str(e),
                    },
                )
                raise

        return wrapper

    return decorator


# Prometheus metric helper functions
def update_memory_count(count: int):
    """Update the total memory count metric"""
    if PROMETHEUS_AVAILABLE:
        METRICS['memories_total'].set(count)


def update_vector_store_size(store_type: str, size: int):
    """Update vector store size metric"""
    if PROMETHEUS_AVAILABLE:
        METRICS['vector_store_size'].labels(store_type=store_type).set(size)


def record_cache_operation(operation: str, cache_type: str, hit: bool):
    """Record cache operation and update metrics"""
    if PROMETHEUS_AVAILABLE:
        result = 'hit' if hit else 'miss'
        METRICS['cache_operations_total'].labels(
            operation=operation,
            cache_type=cache_type,
            result=result
        ).inc()


def update_cache_hit_ratio(cache_type: str, hit_ratio: float):
    """Update cache hit ratio metric"""
    if PROMETHEUS_AVAILABLE:
        METRICS['cache_hit_ratio'].labels(cache_type=cache_type).set(hit_ratio)


def get_prometheus_metrics():
    """Get all Prometheus metrics for export"""
    return METRICS if PROMETHEUS_AVAILABLE else {}
