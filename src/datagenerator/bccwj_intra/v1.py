import json
import pickle
import re
from collections import defaultdict
from typing import Any as NDArray
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def generate(
    dataset: Dict,
):
    tokenizer = AutoTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    X = []
    Y = []
    Z = []
    X_sentence = []
    X_target_verb = []
    Y_ga = []
    Y_wo = []
    Y_ni = []
    loader = DatasetLoading()
    for x, y, z in loader.making_intra_datasets():
        ids, is_target_verb = x
    return samples


class DatasetLoading:
    def __init__(
        self,
        _datasets,
        media=["OC", "OY", "OW", "PB", "PM", "PN"],
    ):
        self.media = media
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )

    def making_intra_datasets(self, datasets):
        for domain in self.media:
            print(f"--- start loading {domain} ---")
            for file in datasets.keys():
                if domain in file:
                    df = datasets[file]
                    for sentential_df in self._to_sentential_df(df):
                        for x, y in self._df_to_intra_vector(sentential_df):
                            yield x, y, file

    def _to_sentential_df(self, df):
        last_sentence_indices = df["is文末"][df["is文末"] == True].index
        start = 0
        for index in last_sentence_indices:
            end = index
            yield df.loc[start:end]
            start = index + 1

    def _case_id_to_index(self, df, case_id, case_type, is_intra):
        if (
            case_type == "none"
            or case_type == "exoX"
            or case_type == "exo2"
            or case_type == "exo1"
        ):
            return str((df["単語"] == "<EOS>").idxmax())
        elif is_intra and case_type == "inter(zero)":
            return str((df["単語"] == "<EOS>").idxmax())
        else:
            return str((df["id"] == case_id).idxmax())

    def _df_to_intra_vector(self, df):
        token = df["単語"].values
        ids = self.tokenizer.convert_tokens_to_ids(token)

        for index, row in df.iterrows():
            if row["verb_type"] == "noun" or row["verb_type"] == "pred":
                y = row.loc[["ga", "ga_type", "o", "o_type", "ni", "ni_type"]].copy()
                cases = ["ga", "o", "ni"]
                for case in cases:
                    case_types = y[f"{case}_type"].split(",")
                    case_ids = y[f"{case}"].split(",")
                    case_indices = []
                    for case_type, case_id in zip(case_types, case_ids):
                        case_index = self._case_id_to_index(
                            df, case_id, case_type, True
                        )
                        case_indices.append(case_index)
                    case_indices = ",".join(case_indices)
                    y[case] = case_indices
                is_target_verb = np.zeros_like(ids)
                is_target_verb[index] = 1
                x = (ids, is_target_verb)
                y = ()
                z = (token, row["verb_type"])
                yield x, y, z
