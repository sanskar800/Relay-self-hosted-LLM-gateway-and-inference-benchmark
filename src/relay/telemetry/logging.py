"""Structured JSON logs: one JSON object per line, with the request id on every line.

Relay's modules keep using the standard `logging` API; this formatter renders their
records (and uvicorn's) as JSON and adds context bound for the current request
(request_id). Prompts and completions are never logged.
"""

import logging
import sys

import structlog

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    shared = [
        structlog.contextvars.merge_contextvars,  # e.g. request_id bound by the middleware
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    relay = logging.getLogger("relay")
    relay.handlers = [handler]
    relay.setLevel(level)
    relay.propagate = False
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level)  # otherwise inherited from root (WARNING): startup lines lost
        logger.propagate = False
    # Relay writes its own access line per request (with tenant, model, tokens, ...).
    logging.getLogger("uvicorn.access").disabled = True

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


access_log = structlog.get_logger("relay.access")
