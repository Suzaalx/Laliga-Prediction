import logging
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
#Consistent logging formatting aids debugging and CI traceability for each pipeline stage.