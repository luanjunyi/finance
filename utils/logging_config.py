import logging

def setup_logging(level=logging.INFO):
    """Configure logging with filename and line number."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
