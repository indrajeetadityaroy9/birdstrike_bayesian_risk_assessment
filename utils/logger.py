import logging

LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(name)s:%(lineno)d]-%(message)s'
DATE_FORMAT = '%H:%M:%S'
_CONFIGURED = False

def configure_logging(level=logging.INFO):
    global _CONFIGURED
    root_logger = logging.getLogger()
    if not _CONFIGURED:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)
        for noisy in ('matplotlib', 'PIL', 'sklearn'):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    root_logger.setLevel(level)

def get_logger(name=None, level=logging.INFO):
    configure_logging(level)
    logger = logging.getLogger(name if name else "birdstrike")
    logger.setLevel(level)
    return logger
