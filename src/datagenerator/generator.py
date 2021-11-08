import argparse
import importlib
import json
import logging
import os
import pickle
from pathlib import Path

# from numpy.typing import DTypeLike, NDArray
from typing import Any as NDArray
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.argparser import create_dataextractor_parser, create_datagenerator_parser
from src.utils.io import get_downstream_dir, get_upstream_dir
from src.utils.types import SampleType

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setStream(tqdm)
_logger.addHandler(_handler)


class Generator:
    def __init__(
        self,
        generator_module: str,
        input_path: Path = get_upstream_dir(),
        output_path: Path = get_downstream_dir(),
    ):
        self._generator_module = generator_module
        self._input_path = input_path
        self._output_path = output_path

    def get(self, dataset: Dict, **kwargs) -> Dict[str, Tuple[List[NDArray]]]:
        module = importlib.import_module(self._generator_module, "src.datagenerator")

        samples = module.generate(dataset, **kwargs)
        return samples

    def load_dataset(self, file: str):
        pickle_path = self._input_path.joinpath(file)
        if not pickle_path.exists():
            raise FileNotFoundError(f"{pickle_path} is not found")
        with open(pickle_path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def save(
        self,
        sample_type: SampleType,
        X: List[NDArray],
        Y: List[NDArray],
        Z: List[NDArray] = None,
    ) -> None:
        filename = sample_type.value + "_X.npz"
        path = self._output_path.joinpath(filename)
        os.makedirs(path.parent, exist_ok=True)
        np.savez_compressed(path, *X)

        filename = sample_type.value + "_Y.npz"
        path = self._output_path.joinpath(filename)
        np.savez_compressed(path, *Y)

        if Z is not None:
            filename = sample_type.value + "_Z.npz"
            path = self._output_path.joinpath(filename)
            np.savez_compressed(path, *Z)


def main():
    dataextractor_parser = create_dataextractor_parser()
    datagenerator_parser = create_datagenerator_parser()
    parser = argparse.ArgumentParser(
        parents=[dataextractor_parser, datagenerator_parser]
    )
    args = parser.parse_args()

    generator_module = args.generator_module
    _logger.info(f"generator: start Generating data: by {generator_module}")
    generator = Generator(generator_module)

    for sample_type, file in zip(SampleType.get_list(), args.files):
        dataset = generator.load_dataset(file=file)

        sample = generator.get(dataset)
        _logger.info(f"generator: start Saving data {sample_type}")
        generator.save(sample_type, *sample)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
