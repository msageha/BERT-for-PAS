from __future__ import annotations

from enum import Enum
from typing import List


class SampleType(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"

    @staticmethod
    def get_list() -> List[SampleType]:
        return [SampleType.TRAIN, SampleType.TEST, SampleType.VALIDATION]
