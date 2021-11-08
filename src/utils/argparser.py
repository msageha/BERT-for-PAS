import argparse
from pathlib import Path

_domain_list = ["OC", "OW", "OY", "PB", "PM", "PN"]


def create_dataextractor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gcs_dir", type=Path)
    parser.add_argument("--files", nargs="+", required=True, type=str)
    return parser


def create_datagenerator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--generator_module", type=str)
    return parser


def create_modeltrainer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_module", required=True, type=str)
    parser.add_argument("--model_weights_path", type=Path)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument(
        "--domains", nargs="+", required=True, choices=_domain_list, type=str
    )
    return parser


def create_modelevaluator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluator_module", required=True, type=str)
    parser.add_argument("--evaluator_params", type=str)
    return parser
