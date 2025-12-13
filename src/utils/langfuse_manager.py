"""LangFuse integration for observability and tracing"""

from typing import Optional, Dict, Any
from functools import wraps
import time
import uuid
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("langfuse_manager")


def generate_txnid() -> str:
    """Generate a unique transaction ID for audit tracking"""
    return f"txn_{uuid.uuid4().hex[:16]}"


# Global LangFuse client
_langfuse_client = None
_langfuse_enabled = False


def initialize_langfuse():
    """Initialize LangFuse client if credentials are available"""
    global _langfuse_client, _langfuse_enabled

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        # Check if credentials are provided
        if (
            Config.LANGFUSE_PUBLIC_KEY
            and Config.LANGFUSE_SECRET_KEY
            and Config.LANGFUSE_PUBLIC_KEY != "pk-lf-your_public_key_here"
            and Config.LANGFUSE_SECRET_KEY != "sk-lf-your_secret_key_here"
        ):
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=Config.LANGFUSE_PUBLIC_KEY,
                secret_key=Config.LANGFUSE_SECRET_KEY,
                host=Config.LANGFUSE_HOST,
            )
            _langfuse_enabled = True
            logger.info(f"LangFuse initialized successfully at {Config.LANGFUSE_HOST}")
        else:
            logger.info(
                "LangFuse credentials not provided - tracing disabled. "
                "Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env to enable tracing."
            )
            _langfuse_enabled = False
    except ImportError:
        logger.warning(
            "LangFuse package not installed - tracing disabled. "
            "Install with: pip install langfuse"
        )
        _langfuse_enabled = False
    except Exception as e:
        logger.error(f"Failed to initialize LangFuse: {e}")
        _langfuse_enabled = False

    return _langfuse_client


def get_langfuse_client():
    """Get or initialize LangFuse client"""
    if _langfuse_client is None:
        initialize_langfuse()
    return _langfuse_client


def is_langfuse_enabled() -> bool:
    """Check if LangFuse tracing is enabled"""
    return _langfuse_enabled


class LangFuseTracer:
    """Context manager for LangFuse tracing"""

    def __init__(
        self,
        name: str,
        trace_type: str = "span",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_trace=None,
        txnid: Optional[str] = None,
    ):
        """
        Initialize tracer

        Args:
            name: Name of the trace
            trace_type: Type of trace ('trace', 'span', 'generation', 'event')
            metadata: Additional metadata
            user_id: User identifier
            session_id: Session identifier
            parent_trace: Parent trace for nested tracing
            txnid: Transaction ID for audit tracking (auto-generated if not provided)
        """
        self.name = name
        self.trace_type = trace_type
        self.metadata = metadata or {}
        self.user_id = user_id
        self.session_id = session_id
        self.parent_trace = parent_trace
        self.txnid = txnid or generate_txnid()
        self.metadata["txnid"] = self.txnid  # Always include txnid in metadata
        self.trace = None
        self.start_time = None

    def __enter__(self):
        """Start tracing"""
        if not is_langfuse_enabled():
            return self

        try:
            client = get_langfuse_client()
            self.start_time = time.time()

            if self.trace_type == "trace":
                self.trace = client.trace(
                    name=self.name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    metadata=self.metadata,
                )
            elif self.trace_type == "span" and self.parent_trace:
                self.trace = self.parent_trace.span(
                    name=self.name, metadata=self.metadata
                )
            elif self.trace_type == "generation" and self.parent_trace:
                self.trace = self.parent_trace.generation(
                    name=self.name, metadata=self.metadata
                )
            elif self.trace_type == "event" and self.parent_trace:
                self.trace = self.parent_trace.event(
                    name=self.name, metadata=self.metadata
                )

        except Exception as e:
            logger.warning(f"Failed to start LangFuse trace '{self.name}': {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracing"""
        if not is_langfuse_enabled() or not self.trace:
            return

        try:
            # Calculate duration
            if self.start_time:
                duration_ms = (time.time() - self.start_time) * 1000
                self.metadata["duration_ms"] = duration_ms

            # Update trace with final metadata
            if hasattr(self.trace, "update"):
                self.trace.update(metadata=self.metadata)

            # Log error if exception occurred
            if exc_type is not None:
                self.metadata["error"] = str(exc_val)
                logger.warning(f"Trace '{self.name}' ended with error: {exc_val}")

        except Exception as e:
            logger.warning(f"Failed to end LangFuse trace '{self.name}': {e}")


def trace_llm_call(operation_name: str):
    """
    Decorator for tracing LLM calls

    Args:
        operation_name: Name of the LLM operation

    Usage:
        @trace_llm_call("intent_classification")
        def classify_intent(self, state):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)

            try:
                # Extract user_id and session_id from state if available
                user_id = None
                session_id = None

                # Check if state is passed as argument
                if args and len(args) > 1 and isinstance(args[1], dict):
                    state = args[1]
                    user_id = state.get("user_id")
                    session_id = state.get("session_id") or state.get("user_id")
                elif kwargs.get("state"):
                    state = kwargs["state"]
                    user_id = state.get("user_id")
                    session_id = state.get("session_id") or state.get("user_id")

                with LangFuseTracer(
                    name=operation_name,
                    trace_type="trace",
                    metadata={"function": func.__name__, "operation": operation_name},
                    user_id=user_id,
                    session_id=session_id,
                ) as tracer:
                    result = func(*args, **kwargs)

                    # Add result metadata if available
                    if tracer.trace and isinstance(result, dict):
                        if "intent" in result:
                            tracer.metadata["intent"] = result["intent"]
                        if "response" in result:
                            tracer.metadata["response_length"] = len(
                                str(result["response"])
                            )

                    return result

            except Exception as e:
                logger.error(f"Error in traced function {operation_name}: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_retrieval(operation_name: str):
    """
    Decorator for tracing retrieval operations

    Args:
        operation_name: Name of the retrieval operation

    Usage:
        @trace_retrieval("policy_retrieval")
        def retrieve_policies(self, query):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)

            try:
                query = kwargs.get("query") or (args[1] if len(args) > 1 else None)

                with LangFuseTracer(
                    name=operation_name,
                    trace_type="trace",
                    metadata={
                        "function": func.__name__,
                        "operation": operation_name,
                        "query": str(query)[:200] if query else None,
                    },
                ) as tracer:
                    result = func(*args, **kwargs)

                    # Add result metadata
                    if tracer.trace:
                        if isinstance(result, list):
                            tracer.metadata["results_count"] = len(result)
                        elif isinstance(result, dict) and "documents" in result:
                            tracer.metadata["results_count"] = len(
                                result.get("documents", [])
                            )

                    return result

            except Exception as e:
                logger.error(f"Error in traced retrieval {operation_name}: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator


def flush_langfuse():
    """Flush LangFuse traces (useful for ensuring all traces are sent before shutdown)"""
    if is_langfuse_enabled() and _langfuse_client:
        try:
            _langfuse_client.flush()
            logger.info("LangFuse traces flushed successfully")
        except Exception as e:
            logger.warning(f"Failed to flush LangFuse traces: {e}")


# Initialize on import
initialize_langfuse()
