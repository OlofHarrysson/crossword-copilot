import logging

logger = logging.getLogger("mycode-logger")

# Create and configure handler
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

# Configure only our logger instance
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Prevent propagation to root logger
logger.propagate = False
