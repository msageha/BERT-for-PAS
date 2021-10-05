import argparse
from pathlib import Path

_loss_function_list = [
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "cosine_similarity",
    "log_cosh",
]

_metrics_list = [
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "cosine_similarity",
    "log_cosh",
]

_optimizer_list = ["SGD", "Adam"]


def create_dataextractor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query", nargs="+", required=True, type=Path)
    return parser


def create_datagenerator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--generator_module", type=str)
    parser.add_argument("--generator_params", type=str)
    return parser


def create_modeltrainer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_module", required=True, type=str)
    parser.add_argument("--model_weights_path", type=Path)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--loss", required=True, choices=_loss_function_list, type=str)
    parser.add_argument("--optimizer", required=True, choices=_optimizer_list, type=str)
    parser.add_argument(
        "--metrics", nargs="+", required=True, choices=_metrics_list, type=str
    )
    parser.add_argument("--model_params", required=True, type=str)
    return parser


def create_modelevaluator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluator_module", required=True, type=str)
    parser.add_argument("--evaluator_params", type=str)
    return parser
