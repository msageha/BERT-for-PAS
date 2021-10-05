import argparse
import logging
import os
from pathlib import Path

# from numpy.typing import DTypeLike
from typing import Any as DTypeLike
from typing import Union

import pandas as pd
from google.oauth2.service_account import Credentials
from tqdm import tqdm

from src.dataextractor.helper import read_query
from src.utils.argparser import create_dataextractor_parser
from src.utils.environ import BQ_PROJECT_ID, CREDENTIALS_PATH
from src.utils.io import get_downstream_dir
from src.utils.visualize import vis_df_head

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setStream(tqdm)
_logger.addHandler(_handler)


def _create_and_save(query_path: Path) -> None:
    credentials = Credentials.from_service_account_file(CREDENTIALS_PATH)
    _logger.info("start Fetching data from bq")
    query = read_query(query_path)
    extractor = Extractor(
        project_id=BQ_PROJECT_ID, credentials=credentials, query=query
    )

    _logger.info(f"start Getting data --- query path: {query_path}")
    df = extractor.get(fillna=0, dtype=None)

    filename = query_path.stem + "_df.pkl"
    _logger.info(f"start Dumping data --- {filename}")
    extractor.save_df(df, filename)

    vis_df_head(df)
    # gcs_path = upload_df_to_gcs_as_csv(query_path, df)
    # _logger.info(f'uploaded to: {gcs_path}')


class Extractor:
    def __init__(
        self,
        project_id: str,
        credentials: Credentials,
        query: str,
        output_path: Path = get_downstream_dir(),
    ):
        self._project_id = project_id
        self._credentials = credentials
        self._query = query
        self._output_path = output_path

    def get(
        self, fillna: Union[float, str] = None, dtype: DTypeLike = None
    ) -> pd.DataFrame:
        """

        Args:
            fillna: specify value of filling None to the DataFrame
            dtype: specify dtype of applying astype func to the DataFrame
                if not specified, not applying astype func

        Returns:

        """
        df = pd.read_gbq(
            self._query,
            project_id=self._project_id,
            credentials=self._credentials,
            dialect="standard",
            progress_bar_type="tqdm",
        )

        if fillna is not None:
            df.fillna(inplace=True, value=fillna)
        if dtype is not None:
            df = df.astype(dtype=dtype, copy=False)
        return df

    def save_df(self, df: pd.DataFrame, filename: str):
        path = self._output_path.joinpath(filename)
        os.makedirs(path.parent, exist_ok=True)
        df.to_pickle(path.as_posix())


def main():
    dataextractor_parser = create_dataextractor_parser()
    parser = argparse.ArgumentParser(parents=[dataextractor_parser])
    args = parser.parse_args()

    _logger.info("Fetching and Saving data from bq")
    for query_path in args.query:
        path = Path("src/dataextractor/queries/").joinpath(query_path)
        _create_and_save(path)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
