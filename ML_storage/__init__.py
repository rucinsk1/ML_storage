import logging
import os
import sys

# log level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "debug").upper()


def get_logger() -> logging.Logger:
    logger = logging.getLogger("ML_storage")
    logger.setLevel(LOG_LEVEL)
    # we need to manually set to stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] <%(name)s> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.handlers = []
    if len(logger.handlers) == 0:
        logger.addHandler(handler)
    return logger
