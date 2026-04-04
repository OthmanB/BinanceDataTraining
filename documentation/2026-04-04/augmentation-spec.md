# Standalone Augmentation Subpackage Specification

**Date:** 2026-04-04
**Version:** 0.1
**Status:** Draft for review

---

## 1. Executive Summary

This document specifies a new standalone `augmentation/` subpackage for `BinanceDataTraining`.

The subpackage is intended to solve a concrete pipeline problem: after coarse class balancing by undersampling, some target classes may still remain below the minimum acceptable sample count. The augmentation subpackage will generate additional semi-synthetic training examples from real market data while preserving strict market-structure constraints, explicit provenance, deterministic replay, simulator-driven validation, and machine-readable diagnostics.

This specification is **end-state-first**. It does not define a throwaway early-phase architecture. The package boundary, data model, simulator contract, diagnostics contract, provenance model, and training integration contract are defined here as stable foundations. Implementation may proceed incrementally, but only against these fixed foundations.

---

## 2. Goals

- Create a **fully independent augmentation subpackage** under `augmentation/`.
- Support semi-synthetic generation of training samples for rare intensity classes.
- Integrate with the existing training pipeline through a **thin adapter**, not by embedding augmentation logic into `training/pipeline.py`.
- Preserve compatibility with the current snapshot-based training flow:
  - two-head intensity labels (`y_up`, `y_down`)
  - snapshot tensors
  - auxiliary features
  - optional long-term features
  - existing normalization pipeline
  - sample weighting using `anchor_ts` and `duty_cycle`
- Provide a **simulator and diagnostics suite** comparable in rigor to `pipelineml-anomaly-injector`.
- Be fully testable with:
  - unit tests
  - property tests
  - scenario tests
  - integration tests
  - reproducible diagnostic artifacts

---

## 3. Non-Goals

- This package is **not** a generic plugin framework.
- This package is **not** a black-box GAN/VAE project in its initial design.
- This package does **not** modify existing snapshot chunk storage as its primary mode of operation.
- This package does **not** blur train/validation/test boundaries.
- This package does **not** silently invent labels without explicit relabeling logic and plausibility validation.

---

## 4. Problem Statement

The current repository already contains an undersampling mechanism in:

- `training/sample_balancing.py`
- `training/pipeline.py`

Today, the effective behavior is:

1. compute available class counts
2. compute undersampled keep counts
3. select retained indices
4. trim to a full-batch-compatible size
5. fail fast if the retained set falls below:
   - `preprocessing.class_balancing.undersampling.min_samples_after_balance`
   - `preprocessing.class_balancing.undersampling.min_fraction_after_balance`

This is architecturally clean but incomplete for severe class imbalance.

The augmentation subpackage must address the missing capability:

> after undersampling has reduced overrepresented classes, fill rare classes with realistic semi-synthetic examples rather than failing or losing too much representativity.

### 4.1 Hard Domain Constraint

This is **not** scalar timeseries augmentation.

Training inputs are order-book-derived tensors. Therefore the augmentation package must operate under market-structure constraints such as:

- non-negative quantities
- valid bid/ask ordering
- valid monotone price ladders
- semantically valid mask channels
- consistency between visible market state and assigned labels

### 4.2 Architectural Constraint

The current persisted snapshot sample contains the visible input window and labels, but not necessarily the full future horizon context required to directly re-derive labels after arbitrary sample mutation.

Therefore the architecture must explicitly distinguish between:

- **label-preserving augmentation** on stored training examples
- **label-changing augmentation** that requires richer context and/or simulation-backed plausibility checks

This distinction is foundational and must be explicit in the package contract.

---

## 5. Architectural Position

The augmentation project will be specified as a **single coherent end-state product** with the following major subsystems designed from the start:

- domain contracts
- config schema and validation
- augmentation engine
- transform registry
- constraint enforcement
- relabeling and plausibility evaluation
- diagnostics
- simulator
- training adapter
- provenance and replay

Implementation may still be incremental, but the architecture must already support the final intended product.

### 5.1 Why End-State-First is Required

The parts most likely to force later rewrites are not individual transforms; they are:

- the canonical augmentation sample model
- legality rules for label changes
- long-term feature policy
- provenance and replay requirements
- simulator contract
- diagnostics/report contract
- the separation between augmentation core and training adapter

If these are not fixed now, later implementation will either fork contracts or force disruptive redesign.

---

## 6. Package Boundary and Responsibilities

The new subpackage will live under:

```text
augmentation/
```

### 6.1 Responsibilities of `augmentation/`

The subpackage owns:

- augmentation data contracts
- augmentation policy resolution
- donor selection logic
- transform execution
- constraint enforcement
- relabeling rules
- plausibility scoring
- simulator execution
- diagnostics/report generation
- provenance and deterministic replay
- config validation for augmentation-specific settings

### 6.2 Responsibilities Outside `augmentation/`

The existing training pipeline remains responsible for:

- loading project config
- preparing snapshot datasets
- computing train/val/test split boundaries
- performing base undersampling
- invoking augmentation through a thin adapter
- constructing final train generators

### 6.3 Adapter Rule

`training/pipeline.py` must remain a **thin consumer** of augmentation outputs.

It may:

- compute deficits
- request augmentation
- receive augmented sample bundles
- continue training

It must **not** implement augmentation internals.

---

## 7. Integration With the Current Repository

### 7.1 Current Integration Point

The augmentation adapter must integrate at the existing post-undersampling decision point in `training/pipeline.py`:

- after retained train indices have been selected
- after they have been trimmed to a full-batch-compatible size
- before failing on `min_samples_after_balance` or `min_fraction_after_balance`

This is the correct hook because:

- undersampling remains the first balancing stage
- augmentation becomes a second balancing stage
- training orchestration remains simple
- the adapter can compute exact deficits from retained real samples

### 7.2 Training Adapter Behavior

The adapter will:

1. inspect retained train indices after undersampling
2. compute deficits against configured thresholds and target label distribution
3. resolve augmentation policy
4. request synthetic examples from `augmentation/`
5. merge real retained examples with synthetic examples
6. recompute effective class statistics
7. re-trim to full batches if necessary
8. continue normal training flow

### 7.3 Split Isolation

Augmentation is allowed only for the **training split**.

It must not affect:

- validation split
- test split
- diagnostics on untouched validation/test data

---

## 8. End-State Module Layout

```text
augmentation/
├── __init__.py
├── contracts.py
├── config.py
├── engine.py
├── registry.py
├── provenance.py
├── adapter_types.py
├── donor_selection.py
├── relabeling.py
├── plausibility.py
├── constraints/
│   ├── __init__.py
│   ├── order_book.py
│   ├── tensor_layout.py
│   └── validation.py
├── transforms/
│   ├── __init__.py
│   ├── base.py
│   ├── label_preserving.py
│   ├── targeted_class_change.py
│   ├── microstructure.py
│   └── noise_models.py
├── diagnostics/
│   ├── __init__.py
│   ├── metrics.py
│   ├── report.py
│   ├── visualizer.py
│   └── artifacts.py
├── simulator/
│   ├── __init__.py
│   ├── runner.py
│   ├── scenarios.py
│   ├── report.py
│   └── fixtures.py
└── adapters/
    ├── __init__.py
    ├── training_pipeline.py
    ├── snapshot_dataset.py
    └── raw_sequence.py
```

---

## 9. Canonical Data Model

The canonical augmentation unit must be frozen now.

### 9.1 Primary Training Example Contract

```python
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import numpy as np


@dataclass(frozen=True)
class LabelPair:
    y_up: int
    y_down: int


@dataclass(frozen=True)
class SampleProvenance:
    source_kind: str           # "real" | "synthetic"
    source_sample_id: str
    donor_sample_id: Optional[str]
    donor_label_pair: Optional[LabelPair]
    transform_id: Optional[str]
    policy_id: Optional[str]
    random_seed: Optional[int]
    replay_token: Optional[str]


@dataclass(frozen=True)
class AugmentationSample:
    x: np.ndarray                     # visible-window snapshot tensor, pre-normalization
    aux_features: np.ndarray
    labels: LabelPair
    anchor_ts_seconds: int
    duty_cycle: float
    long_term_features: Optional[np.ndarray]
    metadata: Mapping[str, Any]
    provenance: SampleProvenance
```

### 9.2 Optional Rich Context Contract

Label-changing augmentation cannot rely only on the visible tensor contract above.

Therefore the architecture must support an optional richer context object:

```python
@dataclass(frozen=True)
class ForwardContext:
    future_observations: Optional[np.ndarray]
    future_timestamps: Optional[np.ndarray]
    derived_microstructure_state: Optional[Mapping[str, np.ndarray]]
    raw_context_metadata: Mapping[str, Any]
```

and a combined bundle:

```python
@dataclass(frozen=True)
class AugmentationBundle:
    sample: AugmentationSample
    forward_context: Optional[ForwardContext]
```
```

This distinction is mandatory.

- Stored snapshot samples may provide only `AugmentationSample`.
- Richer raw-sequence adapters may provide `AugmentationBundle`.

### 9.3 Why This Contract Exists

- `AugmentationSample` is enough for label-preserving transforms.
- `ForwardContext` enables targeted class-changing transforms.
- The same package can therefore support both current integration constraints and future richer workflows without breaking the public model.

---

## 10. Feature Partitioning Rules

Every transform must declare which feature partitions it touches.

### 10.1 Mandatory Partitions

The package must treat the following partitions separately:

- price ladder channels
- quantity/depth channels
- auxiliary engineered features
- mask/confidence/observability channels
- optional long-term feature vectors
- metadata-only fields (`anchor_ts_seconds`, `duty_cycle`, provenance)

### 10.2 Default Mutability Policy

- price ladder channels: mutable only through explicit order-book-aware transforms
- quantity/depth channels: mutable only through explicit order-book-aware transforms
- auxiliary engineered features: recompute preferred; direct mutation discouraged
- mask/confidence channels: **read-only by default**
- long-term features: **read-only by default**
- `anchor_ts_seconds`: immutable except for explicit simulator scenarios
- `duty_cycle`: immutable unless transform explicitly models missingness/observability

### 10.3 Long-Term Feature Rule

Long-term features are derived context, not local microstructure state.

Therefore:

- the default policy is to copy donor long-term features unchanged
- transforms may not change long-term features unless they explicitly declare long-term awareness
- if long-term features are enabled but the transform is not long-term-aware, the package must preserve the donor values and mark this in provenance

---

## 11. Label Semantics and Legality Rules

The repository uses a two-head intensity output:

- `y_up`
- `y_down`

The augmentation package must treat balancing and diagnostics over the **joint label pair**, not only over marginal or coarse views.

### 11.1 Label-Preserving Transform

A transform is label-preserving only if:

- it declares itself as such
- it does not alter the semantic meaning of the sample relative to its current labels
- its diagnostics do not detect plausibility drift beyond configured thresholds

### 11.2 Label-Changing Transform

A transform is label-changing only if:

- it declares an explicit target label objective
- it uses a supported relabeling mechanism
- it passes plausibility and realism gates
- it emits explicit provenance stating the original donor labels and final assigned labels

### 11.3 Forbidden Shortcut

The package must **never** change labels by simple reassignment without an explicit transform-specific relabeling justification.

### 11.4 Supported Relabeling Mechanisms

The spec supports two legitimate relabeling modes:

1. **Forward-context relabeling**
   - available when `ForwardContext` exists
   - relabel using the same binning logic as the production target pipeline

2. **Simulation-backed relabeling**
   - available when no true forward context exists
   - relabel only if the transform passes simulator and plausibility gates defined by policy

---

## 12. Augmentation Policies

An augmentation policy defines:

- what deficits are targeted
- which donor pools are allowed
- which transforms are allowed
- whether label-changing transforms are allowed
- what diagnostics must pass
- what acceptance thresholds apply

### 12.1 Policy Types

The architecture must support at least these policy families:

1. **same-class diversification**
   - rare-class donors only
   - label-preserving only

2. **adjacent-class targeted migration**
   - donors from nearby label-pair neighborhoods
   - label-changing allowed with simulator/plausibility checks

3. **majority-to-rare targeted synthesis**
   - donors from abundant classes
   - strongest diagnostics requirements
   - may be disabled in production until validated

### 12.2 Donor Selection Strategy

The package must support donor selection by:

- joint label-pair membership
- marginal up/down label compatibility
- temporal diversity
- donor reuse caps
- optional regime-aware grouping

### 12.3 Donor Reuse Control

The package must enforce configurable donor reuse limits to reduce memorization risk.

---

## 13. Transform Taxonomy

### 13.1 Label-Preserving Transforms

Examples:

- controlled depth noise remixing
- constrained volume reshaping
- spread micro-perturbation that preserves class semantics
- residual remixing from same-class donor neighborhoods
- minor temporal smoothing or localized jitter

### 13.2 Label-Changing Transforms

Examples:

- target up-intensity enhancement
- target down-intensity enhancement
- spread widening/narrowing linked to future class objective
- microstructure-state drift toward target class bins

### 13.3 Transform Requirements

Every transform must declare:

- `transform_id`
- input contract (`sample_only` or `sample_plus_forward_context`)
- touched feature partitions
- label behavior (`preserving` or `changing`)
- legality constraints
- required diagnostics
- deterministic parameterization from seed

---

## 14. Constraint Enforcement

Constraint enforcement is mandatory and centralized.

### 14.1 Order Book Constraints

At minimum, the package must validate/enforce:

- positive prices where applicable
- non-negative quantities
- valid bid/ask ordering
- monotone bid ladder
- monotone ask ladder
- no negative spread
- valid tensor layout per representation
- mask semantics unchanged unless explicitly modeled

### 14.2 Constraint Engine Behavior

The constraint engine must support:

- reject-and-resample
- deterministic repair when safe
- hard failure when legality cannot be restored

### 14.3 Constraint Reporting

Each augmented sample must record:

- whether repair was applied
- which constraints were violated before repair
- whether the sample passed or failed final legality checks

---

## 15. Plausibility and Realism Gates

The augmentation package must implement gates analogous in spirit to the diagnostics-gated workflow in `pipelineml-anomaly-injector`.

### 15.1 Required Diagnostic Classes

- structural legality checks
- donor concentration checks
- feature-distribution distance checks
- class-pair target attainment checks
- synthetic-vs-real separability checks
- optional regime compatibility checks

### 15.2 Minimum Realism Metrics

The spec requires support for metrics such as:

- spread distribution drift
- top-level depth drift
- imbalance drift
- order-book shape summary drift
- donor reuse histogram
- nearest-neighbor distance to real donor pool
- classifier-based real-vs-synthetic separability score

### 15.3 Acceptance Policy

A policy may only accept synthetic samples that satisfy configured realism thresholds.

---

## 16. Determinism, Provenance, and Replay

Reproducibility is mandatory.

### 16.1 Determinism Rules

- top-level augmentation request must accept an explicit seed
- every synthetic sample must record the seed lineage used to create it
- transforms must be deterministic under identical inputs and seeds

### 16.2 Provenance Fields

Each augmented sample must include provenance for:

- source type (`real` / `synthetic`)
- donor sample id
- donor labels
- transform id
- policy id
- replay token
- whether the sample is label-preserving or label-changing

### 16.3 Replay Token

The replay token must uniquely identify the augmentation recipe so a simulation or diagnostics rerun can reproduce the sample.

---

## 17. Public Python API

The package must expose typed programmatic APIs.

### 17.1 Core Engine API

```python
class AugmentationEngine:
    def augment_batch(self, request: "AugmentationRequest") -> "AugmentationResult":
        ...


@dataclass(frozen=True)
class AugmentationRequest:
    samples: list[AugmentationBundle]
    policy_id: str
    random_seed: int
    target_deficits: dict[tuple[int, int], int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AugmentationResult:
    real_samples: list[AugmentationSample]
    synthetic_samples: list[AugmentationSample]
    diagnostics_report: dict[str, Any]
    acceptance_report: dict[str, Any]
```
```

### 17.2 Simulator API

```python
class AugmentationSimulator:
    def run(self, scenario: "SimulationScenario") -> "SimulationReport":
        ...
```

The simulator must use the same engine and the same transform registry. It must not duplicate transform logic.

### 17.3 Training Adapter API

```python
def augment_after_undersampling(...) -> AugmentationResult:
    ...
```

The adapter returns structured augmentation outputs that the training pipeline can consume without understanding augmentation internals.

---

## 18. Simulator Contract

The subpackage must include an internal simulator similar in spirit to the anomaly injector project.

### 18.1 Simulator Responsibilities

- execute augmentation policies on controlled fixtures and sampled real data
- generate visual diagnostics artifacts
- generate machine-readable reports
- support replay by seed and replay token
- support dry-run and comparison modes

### 18.2 Simulator Output Requirements

The simulator must emit:

- structured JSON-like report data
- per-transform acceptance/rejection counts
- donor coverage and reuse summaries
- per-label-pair coverage summaries
- realism metric summaries
- reproducible visual artifacts

### 18.3 Required Artifact Classes

The spec requires at least:

- class-pair coverage heatmap
- donor reuse histogram
- synthetic-vs-real metric comparison plots
- transform acceptance/failure timelines
- realism summary table

---

## 19. Diagnostics Contract

Diagnostics must be first-class outputs, not incidental logs.

### 19.1 Machine-Readable Outputs

The package must emit structured diagnostics data that can be consumed by:

- unit tests
- scenario tests
- CI
- MLflow or other artifact sinks
- manual review

### 19.2 Human-Readable Artifacts

The package must also emit visual and textual summaries for manual review.

### 19.3 Suggested Artifact Naming

The implementation should standardize artifact paths similarly to the simulator conventions in `pipelineml-anomaly-injector`, for example:

```text
augmentation_diagnostics/
  report.json
  acceptance_summary.json
  class_pair_coverage_heatmap.png
  donor_reuse_histogram.png
  realism_metrics.png
  transform_timeline.png
```

---

## 20. Configuration Schema

The package must have its own top-level config namespace.

### 20.1 Top-Level Namespace

```yaml
augmentation:
  enabled: false
  schema_version: 1
  random_seed: 42

  integration:
    training_after_undersampling:
      enabled: true
      policy_id: "rare_class_fill_default"
      include_synthetics_in_class_weight_stats: true
      include_synthetics_in_effective_train_counts: true
      include_synthetics_in_undersampling_decision: false

  donor_selection:
    target_space: "joint_label_pair"
    max_reuse_per_donor: 3
    time_diversity:
      enabled: true
      min_time_separation_seconds: 3600

  policies:
    rare_class_fill_default:
      mode: "mixed"
      allow_label_changing: true
      allowed_transform_ids:
        - "same_class_depth_noise"
        - "same_class_residual_remix"
        - "targeted_upshift"
        - "targeted_downshift"
      realism_thresholds:
        max_spread_ks: 0.2
        max_depth_drift: 0.15
        max_real_vs_synth_auc: 0.7
      acceptance:
        require_constraints_pass: true
        require_target_label_attainment: true

  simulator:
    enabled: true
    save_artifacts: true
    scenarios:
      - "same_class_diversification"
      - "adjacent_class_targeted"

  diagnostics:
    enabled: true
    save_artifacts: true
    save_reports: true
```
```

### 20.2 Validation Rules

Unknown keys and invalid combinations must fail fast.

Examples:

- label-changing policy with no relabeling mode -> invalid
- transform mutates long-term features without explicit support -> invalid
- synthetic samples included in undersampling decision without provenance-aware policy -> invalid

---

## 21. Training Flow Semantics

The end-state training integration must obey this sequence:

1. prepare dataset
2. compute split boundaries
3. perform base undersampling on real samples only
4. compute deficits from retained real samples
5. call augmentation adapter
6. obtain synthetic samples plus diagnostics/provenance
7. recompute effective train statistics on the merged train set
8. trim merged train source to full batches
9. continue generator construction

### 21.1 Rule on Undersampling Participation

Synthetic samples must **not** participate in the initial undersampling decision unless a future provenance-aware policy explicitly says otherwise.

### 21.2 Rule on Class Weights

Whether synthetic samples contribute to class weight computation must be configurable and explicitly reported.

---

## 22. Stored-Sample vs Rich-Context Modes

This is a core architectural distinction.

### 22.1 Stored-Sample Mode

Inputs available:

- visible tensor `x`
- aux features
- labels
- metadata
- optional long-term features

Supports safely:

- label-preserving augmentation
- constrained same-class diversification
- some simulator-backed label-changing transforms if policy allows

### 22.2 Rich-Context Mode

Inputs additionally available:

- forward horizon context
- raw microstructure state
- future timestamps or equivalent relabeling context

Supports safely:

- explicit targeted class-changing augmentation
- re-derivation of labels using the production target logic

### 22.3 Design Consequence

The package must support both modes through the same contracts and engine, but it must never pretend they have the same relabeling certainty.

---

## 23. Testing Strategy

The subpackage must be testable in isolation.

### 23.1 Unit Tests

Required for:

- config validation
- registry behavior
- donor selection
- transform determinism
- provenance generation
- constraints and repair logic
- diagnostics metric computation

### 23.2 Property Tests

Required for:

- order-book legality invariants
- deterministic replay under equal seeds
- non-negative quantity constraints
- monotone ladder invariants after transformation
- mask-channel preservation where required

The repository already uses `hypothesis`; this package should follow the same property-test style.

### 23.3 Scenario Tests

Required for:

- same-class diversification scenarios
- adjacent-class targeted migration scenarios
- majority-to-rare targeted generation scenarios
- donor reuse limit enforcement
- simulator artifact generation

### 23.4 Thin Integration Tests

Required to verify:

- training adapter can be invoked from the current snapshot pipeline
- merged real + synthetic training source preserves expected generator contract
- class statistics and batch trimming remain consistent

---

## 24. Acceptance Criteria

The augmentation subpackage is not considered production-ready until all of the following are true:

- package compiles/imports cleanly with a stable public API
- config validation is fail-fast and explicit
- label-preserving transforms satisfy legality and determinism tests
- label-changing transforms satisfy relabeling and realism gates
- simulator produces reproducible reports and artifacts
- diagnostics are machine-readable and human-reviewable
- training adapter integration works without contaminating validation/test splits
- provenance is complete enough for replay and audit

---

## 25. Implementation Guidance

This document specifies the **target architecture**, not a forced coding order.

Implementation should proceed incrementally, but only in ways that preserve these fixed contracts.

The preferred implementation order is:

1. contracts/config/registry/provenance
2. constraints and diagnostics foundation
3. simulator/report plumbing
4. label-preserving transforms
5. training adapter integration
6. label-changing transforms with rich diagnostics and replay

This is an implementation sequence, not a different architecture.

---

## 26. Review Checklist

This document should be reviewed specifically for:

- whether `augmentation/` has the correct package boundary
- whether the canonical data model is sufficient
- whether stored-sample vs rich-context distinction is correctly drawn
- whether long-term feature policy is strict enough
- whether relabeling legality is explicit enough
- whether diagnostics and simulator requirements are strong enough
- whether training integration stays thin enough
- whether provenance/replay requirements are sufficient for auditability

---

## 27. Key Architectural Decisions Captured Here

This specification intentionally commits to the following:

- `augmentation/` is a standalone subpackage, not scattered helper logic
- training integration is adapter-based and thin
- the package owns simulator and diagnostics from the start
- the package owns provenance and replay from the start
- balancing is evaluated on joint label pairs, not only coarse classes
- label-changing augmentation is legal only through explicit relabeling modes
- stored-sample and rich-context modes are both supported, but clearly distinguished
- long-term features and mask channels are immutable by default

---

## 28. References

- Current training integration points:
  - `training/pipeline.py`
  - `training/sample_balancing.py`
  - `training/snapshot_dataset.py`
- Current diagnostics patterns in this repository:
  - `diagnostics/snapshot_diagnostics.py`
  - `tests/test_snapshot_diagnostics.py`
- External inspiration for simulator/diagnostics rigor:
  - `/home/obenomar/Work/appthrust/pipelineml-anomaly-injector/simulator/runner.py`
  - `/home/obenomar/Work/appthrust/pipelineml-anomaly-injector/simulator/diagnostic_visualizer.py`
  - `/home/obenomar/Work/appthrust/pipelineml-anomaly-injector/src/pipelineml_anomaly_injector/strategy2/engine.py`
