import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.utils.argparser import create_dataextractor_parser
from src.utils.io import download_from_bucket, get_downstream_dir

# from numpy.typing import DTypeLike


_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setStream(tqdm)
_logger.addHandler(_handler)


class Extractor:
    def __init__(
        self,
        gcs_dir: Path,
        output_path: Path = get_downstream_dir(),
    ):
        self._gcs_dir = gcs_dir
        self._output_path = output_path

    def get(self, file: str):
        """

        Args:
            file: specify value of gcs file path

        Returns:
        """
        gcs_path = self._gcs_dir.joinpath(file)
        file_path = self._output_path.joinpath(file)
        download_from_bucket(file_path, gcs_path)


def main():
    dataextractor_parser = create_dataextractor_parser()
    parser = argparse.ArgumentParser(parents=[dataextractor_parser])
    args = parser.parse_args()
    extractor = Extractor(args.gcs_dir)
    _logger.info("Fetching and Saving data from gcs")
    for file in args.files:
        _logger.info(f"start Getting data --- path: {args.gcs_dir}/{file}")
        extractor.get(file)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
