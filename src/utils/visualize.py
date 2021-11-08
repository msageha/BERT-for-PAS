import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _load_metadata():
    path = Path("/mlpipeline-ui-metadata.json")
    if path.exists():
        with open(path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"outputs": []}
    return metadata


def _dump_metadata(metadata: Dict):
    path = Path("/mlpipeline-ui-metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f)


def vis_df_head(df: pd.DataFrame):
    metadata = _load_metadata()
    output = {
        "type": "table",
        "storage": "inline",
        "format": "csv",
        "header": list(df.columns),
        "source": df.head().to_csv(header=False, index=False),
    }
    metadata["outputs"].append(output)
    _dump_metadata(metadata)


def vis_tensorboard(log_dir: Path):
    metadata = _load_metadata()
    output = {
        "type": "tensorboard",
        "source": log_dir.as_posix(),
    }
    metadata["outputs"].append(output)
    _dump_metadata(metadata)


def vis_html(html: str):
    metadata = _load_metadata()
    output = {"type": "web-app", "storage": "inline", "source": html}
    metadata["outputs"].append(output)
    _dump_metadata(metadata)
