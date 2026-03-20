"""Shared logging configuration for long-running data scripts.

Provides a file + console logging setup used by download_bridge_lidar
and download_and_weak_supervise_hucs (and any future data pipelines).
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(name: str, log_dir: str = './logs') -> logging.Logger:
    """Set up logging to both file and console.

    Creates a timestamped log file under *log_dir* and returns a logger
    that writes INFO+ to the file and WARNING+ to stderr.

    Args:
        name: Logger name and log-file prefix (e.g. 'bridge_processing').
        log_dir: Directory to store log files.

    Returns:
        Configured logger instance.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir_path / f'{name}_{timestamp}.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Log file: %s", log_file)
    print(f"Log file: {log_file}")

    return logger
