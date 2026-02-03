# Binance ML Training Platform - Vision Update (Snapshot Pipeline Alignment)

**Date:** 2026-02-02  
**Time:** 11:45 UTC+09:00  
**Version:** 1.0 - Implementation-aligned update  
**Status:** Active

---

## 1. Executive Summary

This document updates the original vision to reflect the architecture that is now implemented in the repository. The overarching goal remains unchanged: train probabilistic models from high-frequency order book data that capture microstructure patterns and cross-asset effects. The execution path has converged on a snapshot-first pipeline that emphasizes streaming ingestion, reproducible datasets, and explicit handling of missingness across assets.

The snapshot pipeline is now the only supported path in training and evaluation. Instead of a transient in-memory DataObject flow, the system builds persistent snapshot datasets from GreptimeDB in time-ordered chunks. Hybrid depth aggregation compresses raw order book levels into a structured representation that preserves near-market resolution while keeping tensor sizes practical. Multi-asset alignment is applied against a target-asset timeline, and a confidence mask channel is appended to make missingness explicit to the model. Normalization, feature engineering, and evaluation operate directly on snapshot datasets and are logged through MLflow.

This update documents the architecture as built, clarifies how it differs from the original blueprint, and enumerates the remaining work required to reach the broader vision without regressing the current pipeline.

## 2. Why This Update Exists

The original vision in documentation/2025-11-03/vision.md and the 2026-01-17 audit materials were written before the snapshot pipeline existed. As implementation progressed, the team optimized for reliable end-to-end training on real GreptimeDB data and for reproducible dataset generation. The snapshot approach emerged as the most stable path for those goals, and it now serves as the canonical pipeline.

This update preserves the intent of the original design while aligning documentation to the code that is currently deployed. It is not a rewrite of the vision so much as a reconciliation: what is already built is described as the foundation, and the remaining ambitions are framed as extensions that should integrate with the snapshot pipeline rather than replace it.

## 3. Current Architecture (As Built)

The pipeline now follows a snapshot-first flow that keeps memory use bounded and makes dataset production deterministic: GreptimeDB data is streamed in chunks, translated into snapshot records, and persisted as chunked datasets that are keyed by a configuration hash in a manifest. Training and evaluation consume those datasets directly and reuse the same normalization statistics for reproducibility.

In practice, data ingestion and preprocessing are handled in training/snapshot_dataset.py and data/greptime_client.py, with dataset management and hashing in training/snapshot_store.py. Training is orchestrated by training/pipeline.py, and evaluation is performed by evaluation/evaluator.py. The legacy in-memory path is intentionally disabled to avoid divergence from the snapshot dataset logic.

Three practical outcomes define the current architecture. First, the system scales to large time ranges by streaming data chunk-by-chunk instead of loading an entire window into memory. Second, datasets are reproducible because the snapshot manifest stores a deterministic configuration hash. Third, missingness is modeled explicitly in multi-asset inputs through a confidence mask channel that is appended alongside asset channels.

## 4. Data and Preprocessing Strategy

GreptimeDB data is fetched in fixed time chunks using configured request timeouts and retry behavior. Chunked ingestion prevents monolithic queries and provides natural boundaries for dataset persistence. Each chunk is translated into snapshot records at the configured cadence, and per-asset gap handling is applied before alignment. Large gaps in the target asset trigger buffer resets so that samples never straddle discontinuities.

Order book representation uses the hybrid format when representation is set to "hybrid". In this representation, near-market levels are kept at full resolution while deeper levels are aggregated into bins defined by configuration. This hybrid tensor is treated as the canonical representation for depth-aware models, while top-of-book remains available as a lightweight alternative when computation or storage are constrained.

Multi-asset support is implemented through time alignment against the target asset timeline. Alignment can interpolate or bucket snapshots, with configurable tolerance and a missing-policy for large gaps. When missingness exceeds the large-gap threshold, correlated assets are zero-padded by default, while the target asset triggers a gap reset. A confidence mask channel is appended alongside asset channels so that the model can learn the reliability of each aligned input rather than inferring it implicitly.

Normalization is computed and applied within the snapshot pipeline, and mask channels are explicitly excluded from statistics. Normalization stats are stored alongside the snapshot dataset and reused by evaluation to ensure consistency. Feature engineering is implemented in preprocessing/feature_engineering.py and applied in the snapshot builder when enabled, keeping derived features consistent across training and evaluation.

Temporal features remain available and can be concatenated as additional channels when configured. This preserves the original intent of encoding time-of-day and day-of-week effects while staying within the snapshot data flow.

## 5. Model Training and Evaluation

The CNN+LSTM architecture remains the canonical model family. In snapshot mode, the training pipeline supports two-head intensity output aligned to the configured price class boundaries. Training proceeds with chronological splits, optional sample weighting, and MLflow logging of metrics and artifacts.

Evaluation is snapshot-aware and uses the same normalization stats as training. Calibration analysis is available, but post-hoc calibration methods are not yet implemented.

Hyperparameter optimization is currently disabled in snapshot mode to avoid conflicts with snapshot dataset handling and MLflow lifecycle. This constraint is intentional and should be revisited only when the snapshot pipeline can support reproducible tuning runs.

## 6. What Remains to Reach the Full Vision

Several components of the original vision are not yet implemented in the snapshot pipeline and should be treated as extensions rather than replacements. The most significant gap is the absence of the long-term channel and dual-branch model architecture. The current pipeline is optimized for short-term order book dynamics, but it does not yet provide a regime-aware long-term context or multi-scale temporal inputs.

Post-hoc calibration is another missing element. While calibration metrics are computed, temperature scaling or isotonic calibration is not yet available, and probability reliability remains uncorrected. Temporal degradation testing and rolling window validation are likewise not implemented, which means the assumptions about non-stationarity are not yet empirically validated.

Fine-tuning support, backtesting, and automated class weight computation for the two-head output remain open items. These features were part of the original roadmap, but they must be designed to work with snapshot datasets and the current multi-asset alignment logic.

Finally, the MLflow experiment hierarchy described in the original vision has not been implemented. The current system logs activity under a single experiment, which is sufficient for training runs but falls short of the multi-stage lineage design. Any future expansion in this direction should reuse snapshot manifests as the authoritative dataset identity.

## 7. Operational Notes and Known Constraints

The MLflow experiment tracker currently changes the working directory to the configured local artifact path. This simplifies artifact handling, but it means that relative paths in configuration are resolved relative to that directory. To avoid surprises, snapshot.directory and other file paths should be configured as absolute paths until the tracker behavior is revisited.

The snapshot pipeline enforces strict configuration validation. Alignment settings are included in the snapshot hash, so any change in alignment method or thresholds produces a new snapshot dataset by design. This behavior is intentional and should be preserved to maintain reproducibility across experiments.

## 8. Document Lineage and References

This update is intended to be read together with the original vision in documentation/2025-11-03/vision.md and the audit materials in documentation/2026-01-17/code_audit_report.md, documentation/2026-01-17/implementation_priority_matrix.md, and documentation/2026-01-17/technical_debt_register.md. The implementation-aligned planning updates that correspond to this vision are provided in documentation/2025-11-03/implementation_priority_matrix_update_2026-02-02.md and documentation/2025-11-03/technical_debt_register_update_2026-02-02.md.

## 9. Change Log

Version 1.0 (2026-02-02) documents the snapshot-first architecture, multi-asset alignment strategy, and mask-aware normalization as implemented. It also records the remaining roadmap items that must be integrated into the snapshot pipeline to achieve the broader goals described in the original vision.
