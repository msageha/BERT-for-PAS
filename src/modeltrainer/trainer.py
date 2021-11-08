import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# from numpy.typing import NDArray
from tensorflow import data as tf_data
from tensorflow.keras import callbacks, optimizers
from tqdm import tqdm

from src.utils.argparser import create_modeltrainer_parser
from src.utils.io import get_downstream_dir, get_upstream_dir
from src.utils.types import SampleType
from src.utils.visualize import vis_tensorboard

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setStream(tqdm)
_logger.addHandler(_handler)


class Trainer:
    _model = None
    # Because early stopping is applied, epochs is set to a large size by default.
    _EPOCHS = 1000
    # Note that the larger the value, the more memory is used.
    _BUFFER_SIZE = 100000

    def __init__(
        self,
        model_module: str,
        batch_size: int,
        loss: str,
        metrics: List[str],
        optimizer: str,
        learning_rate: float,
        model_params: Dict,
        input_path: Path = get_upstream_dir(),
        output_path: Path = get_downstream_dir(),
    ):
        self._model_module = model_module
        self._batch_size = batch_size
        self._loss = loss
        self._metrics = metrics
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._model_params = model_params
        self._input_path = input_path
        self._output_path = output_path

    def _build_model(self):
        model_module = importlib.import_module(
            self._model_module, "src.modeltrainer.model"
        )
        model = model_module.Model(**self._model_params)
        self._model = model

    def _load_optimizer(self):

        if self._optimizer not in optimizers.__dict__.keys():
            raise NameError(f"{self._optimizer} is incorrect")
        Optimizer = getattr(tf.keras.optimizers, self._optimizer)

        return Optimizer(learning_rate=self._learning_rate)

    def _scheduler(self, epoch, lr):
        if epoch >= 10:
            lr = lr * tf.math.exp(-0.1)
        return lr

    def _create_callbacks(self) -> List[callbacks.Callback]:
        checkpoint_path = self._output_path.joinpath("checkpoint/checkpoint")
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, save_best_only=True
        )

        log_path = self._output_path.joinpath("logs")
        tensorboard = callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=15, mode="min"
        )

        lr_scheduler = callbacks.LearningRateScheduler(self._scheduler)
        return [checkpoint, tensorboard, early_stopping, lr_scheduler]

    def build_and_compile_model(self):
        self._build_model()
        optimizer = self._load_optimizer()
        self._model.compile(loss=self._loss, optimizer=optimizer, metrics=self._metrics)

    def _try_run_model(self, train: tf_data.Dataset):
        example_input = [i[0] for i in list(train.take(1))][0]
        outputs = self._model(example_input)
        return outputs

    def model_summary(self, train, print_fn=None):
        # for the input initialize
        self._try_run_model(train)
        self._model.summary(print_fn=print_fn)

    def _load_best_params(self):
        checkpoint_path = self._output_path.joinpath("checkpoint/checkpoint")
        self.load_model(path=checkpoint_path, from_checkpoint=True)

    def fit_model(self, train: tf_data.Dataset, validation: tf_data.Dataset):
        callbacks = self._create_callbacks()
        history = self._model.fit(
            train, epochs=self._EPOCHS, validation_data=validation, callbacks=callbacks
        )
        self._load_best_params()
        return history

    def evaluate_model(self, test: tf_data.Dataset) -> Dict[str, List[dict]]:
        values = self._model.evaluate(test)
        results = {"metrics": []}
        for name, value in zip(self._model.metrics_names, values):
            result = {
                "name": name,
                "numberValue": value,
            }
            results["metrics"].append(result)
        path = self._output_path.joinpath("metrics-results.json")
        with open(path, "w") as f:
            json.dump(results, f)
        return results

    def save_model(self) -> None:
        path = self._output_path.joinpath("model")
        self._model.save(path, save_format="tf")

    def load_model(self, path: Path, from_checkpoint: bool = True):
        if from_checkpoint:
            self._model.load_weights(path)
        else:
            self._model = tf.keras.models.load_model(path)

    def load_datasets(self) -> Tuple[tf_data.Dataset, tf_data.Dataset, tf_data.Dataset]:
        train = self._load_dataset(SampleType.TRAIN)
        validation = self._load_dataset(SampleType.VALIDATION)
        test = self._load_dataset(SampleType.TEST)

        return train, validation, test

    def _load_dataset(self, sample_type: SampleType) -> tf_data.Dataset:
        filename = sample_type.value + "_X.npz"
        path = Path(self._input_path).joinpath(filename)
        X = np.load(path.as_posix(), allow_pickle=True)
        filename = sample_type.value + "_Y.npz"
        path = Path(self._input_path).joinpath(filename)
        Y = np.load(path.as_posix(), allow_pickle=True)

        X = self._npzfile_to_tf(X)
        Y = self._npzfile_to_tf(Y)
        dataset = tf_data.Dataset.zip((X, Y))
        if sample_type == SampleType.TEST:
            dataset = dataset.batch(self._batch_size, drop_remainder=True)
        else:
            dataset = dataset.shuffle(self._BUFFER_SIZE).batch(
                self._batch_size, drop_remainder=True
            )

        return dataset

    @staticmethod
    def _npzfile_to_tf(X) -> tf_data.Dataset:
        X = tuple(tf.constant(X[arr]) for arr in X.files)
        X = tf_data.Dataset.from_tensor_slices(X)
        return X


def main():
    modeltrainer_parser = create_modeltrainer_parser()
    parser = argparse.ArgumentParser(parents=[modeltrainer_parser])
    args = parser.parse_args()

    _logger.info("start Trainer")
    trainer = Trainer(
        model_module=args.model_module,
        batch_size=args.batch_size,
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        metrics=args.metrics,
        model_params=json.loads(args.model_params),
    )
    _logger.info("start Loading datasets")
    train, validation, test = trainer.load_datasets()

    _logger.info("start Building model")
    trainer.build_and_compile_model()
    if args.model_weights_path:
        _logger.info("start Loading weights")
        path = get_upstream_dir().joinpath(args.model_weights_path)
        trainer.load_model(path, from_checkpoint=True)
    trainer.model_summary(train, print_fn=_logger.info)

    _logger.info("start Fitting model")
    trainer.fit_model(train, validation)

    _logger.info("start Evaluating model")
    trainer.evaluate_model(test)

    _logger.info("start Saving model")
    trainer.save_model()

    log_dir = get_downstream_dir().joinpath("logs")
    vis_tensorboard(log_dir=log_dir)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
