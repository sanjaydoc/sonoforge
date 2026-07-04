"""Data layer: target ingestion, featurization, and the Candidate schema."""

from sonoforge.data.dataset import (
    build_dataset,
    dedup_by_sequence,
    filter_by_length,
    load_candidates,
    records_to_candidates,
    save_candidates,
)
from sonoforge.data.featurize import SequenceFeaturizer, one_hot
from sonoforge.data.fetch import SequenceRecord, fetch_targets, fetch_uniprot
from sonoforge.data.types import Candidate, PropertyRecord

__all__ = [
    "Candidate",
    "PropertyRecord",
    "SequenceFeaturizer",
    "SequenceRecord",
    "build_dataset",
    "dedup_by_sequence",
    "fetch_targets",
    "fetch_uniprot",
    "filter_by_length",
    "load_candidates",
    "one_hot",
    "records_to_candidates",
    "save_candidates",
]
