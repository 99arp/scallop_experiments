from __future__ import annotations

from .models import EventCalculusConfig, EventCalculusResult, ProbabilisticFact
from .tensor_engine import TensorEventCalculusEngine
from .Tensor_pEC_files import dg, readDefinitions


class LiveTensorEventCalculus:
    """In-memory live Tensor-pEC processor for ROS samples."""

    def __init__(self, config: EventCalculusConfig | None = None):
        self.config = config or EventCalculusConfig()
        self.engine = TensorEventCalculusEngine(config=self.config)
        self.facts: list[ProbabilisticFact] = []
        self.caching_order, self.definitions, self.tensors_dim = readDefinitions()
        self.grounding: tuple[tuple[str, str], ...] = ()

    def feed(self, facts: tuple[ProbabilisticFact, ...]) -> EventCalculusResult:
        self.facts.extend(facts)
        narrative = tuple(self.facts)
        self.grounding = dg(narrative)
        return self.engine.infer(narrative)
