import logging

def setup_logging():
    """Configure logging with filename and line number."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
