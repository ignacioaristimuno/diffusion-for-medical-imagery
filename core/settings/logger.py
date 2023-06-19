from logging import getLogger, Formatter, StreamHandler, DEBUG
from logging import Logger


def custom_logger(logger_name: str) -> Logger:
    """Function for returning a Logger object with specified settings"""

    logger = getLogger(f"{logger_name} - ")
    logger.setLevel(DEBUG)
    if not logger.hasHandlers():
        console_handler = StreamHandler()
        console_handler.setLevel(DEBUG)
        formatter = Formatter(
            "%(asctime)s - %(name)s%(levelname)s: %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S%p",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger
