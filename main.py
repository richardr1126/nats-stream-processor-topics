"""Main entry point for NATS Topic Processor service."""

import argparse
import asyncio
import sys
from typing import Optional

from src.config import settings
from src.logging_setup import configure_logging, get_logger
from src.service import TopicProcessorService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NATS Topic Processor Service")
    parser.add_argument("--log-level", default=None, help="Override log level")
    parser.add_argument("--log-format", default=None, choices=("json", "console"), help="Override log format")
    return parser.parse_args()


async def run_service() -> None:
    """Construct and run the TopicProcessorService instance until termination."""
    svc = TopicProcessorService()
    await svc.run()


def main(argv: Optional[list] = None) -> int:
    """Run the async service and return an exit code."""
    args = _parse_args()
    log_level = args.log_level or settings.LOG_LEVEL
    log_format = args.log_format or settings.LOG_FORMAT

    # Configure structured logging as early as possible
    configure_logging(log_level, log_format)
    log = get_logger(__name__)

    try:
        log.info("starting_service", service=settings.SERVICE_NAME)
        asyncio.run(run_service())
        log.info("service_exited", service=settings.SERVICE_NAME)
        return 0
    except KeyboardInterrupt:
        # Allow Ctrl-C to exit cleanly without a stack trace
        log.info("service_interrupted", service=settings.SERVICE_NAME)
        return 0
    except Exception:
        # Log full traceback for unexpected failures and return non-zero exit code
        log.exception("service_failed", service=settings.SERVICE_NAME)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
