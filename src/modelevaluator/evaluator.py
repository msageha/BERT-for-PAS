import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any as NDArray
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

from src.utils.argparser import create_modelevaluator_parser
from src.utils.io import get_downstream_dir, get_upstream_dir
from src.utils.types import SampleType

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setStream(tqdm)
_logger.addHandler(_handler)


class Evaluator:
    _model = None
    _BATCH_SIZE = 128

    def __init__(
        self,
        evaluator_module: str,
        model_path: Union[Path, None] = None,
        input_path: Path = get_upstream_dir(),
        output_path: Path = get_downstream_dir(),
    ):
        self._evaluator_module = evaluator_module
        if model_path is None:
            model_path = input_path.joinpath("model")
        self._model_path = model_path
        self._input_path = input_path
        self._output_path = output_path

    def load_model(self):
        self._model = tf.keras.models.load_model(self._model_path)

    def load_dataset(
        self, sample_type: SampleType
    ) -> Union[Tuple[tf_data.Dataset, tf_data.Dataset, NDArray], None]:
        filename = sample_type.value + "_X.npz"
        path = Path(self._input_path).joinpath(filename)
        if not path.exists():
            return None
        X = np.load(path.as_posix(), allow_pickle=True)

        filename = sample_type.value + "_Y.npz"
        path = Path(self._input_path).joinpath(filename)
        if not path.exists():
            return None
        Y = np.load(path.as_posix(), allow_pickle=True)

        filename = sample_type.value + "_Z.npz"
        path = Path(self._input_path).joinpath(filename)
        if path.exists():
            Z = np.load(path.as_posix(), allow_pickle=True)
        else:
            Z = None

        X = self._npzfile_to_tf(X)
        Y = self._npzfile_to_tf(Y)
        if Z is not None:
            Z = self._npzfile_to_numpy(Z)
        return X, Y, Z

    @staticmethod
    def _npzfile_to_tf(X) -> tf_data.Dataset:
        X = tuple(tf.constant(X[arr]) for arr in X.files)
        X = tf_data.Dataset.from_tensor_slices(X)
        return X

    @staticmethod
    def _npzfile_to_numpy(X) -> NDArray:
        X = tuple(X[arr] for arr in X.files)
        return X

    def evaluate(
        self,
        sample_type: SampleType,
        X: tf_data.Dataset,
        Y: tf_data.Dataset,
        Z: NDArray = None,
        **kwargs,
    ) -> None:
        evaluator_module = importlib.import_module(
            self._evaluator_module, "src.modelevaluator"
        )
        evaluator_module.evaluate(
            model=self._model,
            X=X,
            Y=Y,
            Z=Z,
            output_path=self._output_path.joinpath(sample_type.value),
            **kwargs,
        )


def main():
    modelevaluator_parser = create_modelevaluator_parser()
    parser = argparse.ArgumentParser(parents=[modelevaluator_parser])
    args = parser.parse_args()

    _logger.info("start Evaluator")
    evaluator = Evaluator(args.evaluator_module)
    _logger.info("start Loading model")
    evaluator.load_model()
    for sample_type in SampleType.get_list():
        _logger.info(f"{sample_type} evaluator: start Loading data")
        dataset = evaluator.load_dataset(sample_type)
        if dataset is not None:
            _logger.info(f"{sample_type} evaluator: start Evaluating")
            evaluator.evaluate(
                sample_type, *dataset, **json.loads(args.evaluator_params)
            )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
