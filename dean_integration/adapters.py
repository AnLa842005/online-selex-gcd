"""Protocol and evaluator contracts, independent of the external DEAN core."""
from dataclasses import dataclass
from typing import Protocol

from .config import OFFICIAL_ARGS


class OfficialProtocolAdapter:
    @staticmethod
    def arguments(config, seed, checkpoint):
        if config["mode"] != "official":
            raise ValueError("Official runner cannot execute SSB adaptation")
        if seed not in config["seeds"]:
            raise ValueError("Seed is outside the official sweep")
        return OFFICIAL_ARGS + ["--seed", str(seed), "--exp", "DEAN_Repro",
                                "--vit_pretrained_dino", str(checkpoint), "--LOG_DIR", "./logs"]


@dataclass(frozen=True)
class LearnerBatch:
    """Future strict adapter input. Images/features resolved from opaque IDs."""
    sample_ids: tuple[str, ...]
    inputs: object
    session_id: int


class StrictLearner(Protocol):
    def observe(self, batch: LearnerBatch) -> object: ...


class SSB100ProtocolAdapter:
    def __init__(self, config):
        if config["mode"] != "ssb100_draft":
            raise ValueError("SSB adapter requires a separate draft config")

    def run(self):
        raise NotImplementedError("SSB adaptation is NOT RUN; leakage/exposure audit required")


class EvaluationAdapter(Protocol):
    """Evaluator owns labels/mappings; it receives predictions, never the learner."""
    def evaluate(self, sample_ids: tuple[str, ...], predictions: object) -> dict: ...
