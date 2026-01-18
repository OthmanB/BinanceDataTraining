"""Preprocessing module for BinanceDataTraining.

Exports preprocessing utilities for data transformation, normalization,
depth aggregation, and tensor construction.
"""

from preprocessing.normalizer import Normalizer, create_normalizer_from_config
from preprocessing.depth_aggregator import (
    validate_hybrid_config,
    get_hybrid_output_shape,
    compute_bin_boundaries,
    aggregate_depth_levels,
    aggregate_snapshot_to_hybrid,
)
from preprocessing.snapshot_sequence_builder import (
    build_top_of_book_sequence_tensor,
    build_hybrid_depth_sequence_tensor,
)

__all__ = [
    "Normalizer",
    "create_normalizer_from_config",
    "validate_hybrid_config",
    "get_hybrid_output_shape",
    "compute_bin_boundaries",
    "aggregate_depth_levels",
    "aggregate_snapshot_to_hybrid",
    "build_top_of_book_sequence_tensor",
    "build_hybrid_depth_sequence_tensor",
]
