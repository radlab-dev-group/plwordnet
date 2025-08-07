import sys
import logging

from typing import Optional


def prepare_logger(
    logger_name: Optional[str],
    logger_file_name: Optional[str],
    use_default_config: bool = False,
    log_level: str = "INFO",
) -> logging.Logger:
    """
    Creates and configures a logger instance for the Polish Wordnet application.

    This function sets up a logger with optional basic configuration that outputs
    to both a console (stdout) and a log file. If the basic configuration
    is not applied, the logger uses the existing logging configuration.

    Args:
        logger_name (Optional[str]): Name for the logger instance
        (defaults to "plwordnet")
        logger_file_name (Optional[str]): Name of the log file
        (defaults to "plwordnet.log")
        use_default_config (bool): Whether to apply basic logging
        configuration with INFO level and predefined format (default: False)
        log_level (Optional[str]): Logging level

    Returns:
        logging.Logger: Configured logger instance ready for use

    Note:
        When use_default_config is True, the logger will output
        timestamped messages with logger name and level information
        to both console and file simultaneously.
    """

    l_file_name = (
        logger_file_name if logger_file_name is not None else "plwordnet.log"
    )
    if use_default_config:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(l_file_name),
            ],
        )
    l_name = logger_name if logger_name is not None else "plwordnet"

    logger = logging.getLogger(l_name)
    logger.setLevel(getattr(logging, log_level))

    return logger
