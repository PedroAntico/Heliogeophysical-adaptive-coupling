import logging
from pathlib import Path

def setup_logging(name: str = "HAC", level: str = "INFO"):
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/hac.log")
        ]
    )
    return logging.getLogger(name)
