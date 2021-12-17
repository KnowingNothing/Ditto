import logging


ditto_logger = logging.getLogger("Ditto")
ditto_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(name)s] %(asctime)s: [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
ditto_logger.addHandler(ch)
