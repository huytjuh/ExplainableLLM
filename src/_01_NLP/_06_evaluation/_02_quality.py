from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Coherence:
    c_v: float
    u_mass: float
    c_npmi: float

@dataclass(frozen=True)
class Diversity:
    diversity: float
    uniqueness: float

@dataclass(frozen=True)
class ClusterMetrics:
    similarity: float
    separation: float

    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float

@dataclass(frozen=True)
class OperationalMetrics:
    fallback_rate: float
    entropy: float
    low_confidence_rate: float
    average_confidence: float

    coverage: float

@dataclass
class OperationalConfig:
    threshold: float



class QualityMetrics:
    """Quality metrics for a text and topics."""

    def __init__(self, coherence: Coherence, diversity: Diversity, entropy: Entropy, cluster_metrics: ClusterMetrics) -> None:
        self.coherence = coherence
        self.diversity = diversity
        self.entropy = entropy
        self.cluster_metrics = cluster_metrics

