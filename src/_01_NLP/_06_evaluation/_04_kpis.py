from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class IntentRouting:
    pass

@dataclass(frozen=True)
class TopicModeling:
    pass

@dataclass(frozen=True)
class ConversationalAnalytics:
    pass

@dataclass(frozen=True)
class Sentiment:
    pass

@dataclass(frozen=True)
class EntityExtraction:
    pass

@dataclass(frozen=True)
class Summarization:
    pass
