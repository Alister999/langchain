import logging
import sys
import colorlog

def setup_logging():
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(handler)