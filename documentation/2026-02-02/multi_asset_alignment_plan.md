# Multi-Asset Alignment Plan (Snapshot Pipeline)

**Date:** 2026-02-02

## Goals
- Support multi-asset inputs in the snapshot pipeline with configurable alignment.
- Default to time interpolation on the post-aggregation order book tensor.
- Provide an alternate bucket-based alignment method.
- Keep alignment settings fully config-driven and included in snapshot hashing.

## Scope
- Snapshot pipeline only.
- Legacy in-memory pipeline remains disabled.
- Applies to both top-of-book and hybrid representations.

## Configuration Changes
Add a new block under `data.asset_pairs.alignment`:

```yaml
data:
  asset_pairs:
    alignment:
      method: "interpolate"  # "interpolate" or "bucket"
      missing_policy: "forward_fill"  # "forward_fill", "skip", or "error"
      max_gap_seconds: 60  # Upper bound for interpolation/forward fill
      bucket_tolerance_seconds: 2  # Used only when method="bucket"
```

Notes:
- `method` is defaulted to `"interpolate"` in the reference config.
- `max_gap_seconds` should default to `data.validation.max_gap_seconds` if not set.
- `bucket_tolerance_seconds` is only relevant for `method="bucket"`.
- Alignment settings must be included in the snapshot config hash.

## Alignment Strategy A (Default): Interpolate
**Definition**
- Build per-asset tensors in the snapshot builder (post-aggregation).
- Use the target asset timeline (after gap handling) as the reference.
- Interpolate the entire tensor over the time axis (multidimensional interpolation).

**Implementation detail**
- Build per-asset tensors with shape:
  - Hybrid: `(T, L, 4)` per asset
  - Top-of-book: `(T, H, W)` per asset
- Flatten spatial axes to `(T, F)` and interpolate each feature column over time.
- Reconstruct the tensor to original shape after interpolation.
- Apply `missing_policy` for gaps larger than `max_gap_seconds`.

**Missing policies**
- `forward_fill`: fill until `max_gap_seconds`; otherwise skip.
- `skip`: drop samples with large gaps.
- `error`: raise if any gap exceeds threshold.

## Alignment Strategy B: Bucket
**Definition**
- Bucket timestamps into cadence windows (floor by cadence seconds).
- Align assets by bucket key with tolerance `bucket_tolerance_seconds`.
- Select closest snapshot inside each bucket.

**Missing policies**
Same as Strategy A.

## Snapshot Builder Changes
- Replace strict timestamp equality in `_build_multi_asset_records` with aligned mapping.
- Use alignment method to map correlated asset data onto target asset timeline.
- Preserve target asset timestamps and labels.

## Tests
Add or update unit tests:
- Interpolation alignment on a small synthetic tensor.
- Bucket alignment selection with tolerance.
- Missing data handling for each `missing_policy`.

## Smoke Test Plan
1) Run snapshot training + evaluation with `correlated_assets=["ETHUSDT"]` and default alignment.
2) Repeat with `method="bucket"` and a small tolerance.
3) Confirm snapshot files are generated, training runs, and evaluation logs to MLflow.

## Open Questions (Defaults Suggested)
- Missing policy default: `forward_fill` with `max_gap_seconds`.
- Quantities vs prices interpolation: use linear interpolation for all tensor values (post-aggregation).
