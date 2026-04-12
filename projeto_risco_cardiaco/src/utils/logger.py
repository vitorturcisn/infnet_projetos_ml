import logging
import sys

def get_logger(name: str, logging_config: dict) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = logging_config.get("level", "INFO")
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            datefmt=logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
        )
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger