from loguru import logger
import sys
from src.utils.paths import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<cyan>{extra[module]}</cyan> | "
    "<yellow>{file}:{line}</yellow> | "
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level="INFO",
    enqueue=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    LOGS_DIR / "app.log",
    format=LOG_FORMAT,
    level="INFO",
    rotation="5 MB",
    retention=5,
    enqueue=True,
    backtrace=False,
    diagnose=False,
)

def get_logger(module_name:str):
    return logger.bind(module=module_name)