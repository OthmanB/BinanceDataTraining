"""Property-based tests for snapshot alignment and masking."""

import os
import tempfile
import unittest

import numpy as np

try:
    from hypothesis import HealthCheck, assume, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

import training.snapshot_dataset as sd


_CI_MODE = os.environ.get("CI", "").lower() in ("true", "1", "yes")
_MAX_EXAMPLES = 10 if _CI_MODE else 100

_BASE_TS = np.datetime64("2024-01-01T00:00:00")


def _make_record(
    offset_s: int,
    features: np.ndarray,
    *,
    confidence: float = 1.0,
    gap_reset: bool = False,
) -> sd.SnapshotRecord:
    ts = _BASE_TS + np.timedelta64(int(offset_s), "s")
    bid = float(features[0])
    ask = float(features[2]) if len(features) > 2 else 0.0
    mid_price = 0.5 * (bid + ask) if bid > 0 and ask > 0 else 0.0
    return sd.SnapshotRecord(
        timestamp=ts,
        snapshot_features=[float(x) for x in features[:4]],
        depth=None,
        mid_price=mid_price,
        hybrid_snapshot=None,
        volume_proxy=0.0,
        confidence=float(confidence),
        gap_reset=bool(gap_reset),
    )


def _make_hybrid_record(
    offset_s: int,
    hybrid_snapshot: np.ndarray,
    *,
    confidence: float = 1.0,
) -> sd.SnapshotRecord:
    ts = _BASE_TS + np.timedelta64(int(offset_s), "s")
    snap = np.asarray(hybrid_snapshot, dtype="float32")
    bid = float(snap[0, 0]) if snap.size else 0.0
    ask = float(snap[0, 2]) if snap.size else 0.0
    mid_price = 0.5 * (bid + ask) if bid > 0 and ask > 0 else 0.0
    snapshot_features = [float(snap[0, 0]), float(snap[0, 1]), float(snap[0, 2]), float(snap[0, 3])]
    return sd.SnapshotRecord(
        timestamp=ts,
        snapshot_features=snapshot_features,
        depth=None,
        mid_price=mid_price,
        hybrid_snapshot=snap,
        volume_proxy=0.0,
        confidence=float(confidence),
        gap_reset=False,
    )


def _synthetic_quadratic_features(t_s: np.ndarray | float) -> np.ndarray:
    t = np.asarray(t_s, dtype="float64")
    base = 100.0 + 1e-4 * t**2
    spread = 0.05
    bid = base - spread / 2.0
    ask = base + spread / 2.0
    bid_qty = 5.0 + 1e-5 * t**2
    ask_qty = 4.0 + 1e-5 * t**2
    return np.stack([bid, bid_qty, ask, ask_qty], axis=-1)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestSnapshotAlignmentProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        gap=st.integers(min_value=1, max_value=120),
        offset=st.integers(min_value=0, max_value=120),
        left_vals=st.lists(
            st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
        right_vals=st.lists(
            st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
        left_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        right_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_interpolate_top_of_book_linear(
        self,
        gap: int,
        offset: int,
        left_vals: list[float],
        right_vals: list[float],
        left_conf: float,
        right_conf: float,
    ) -> None:
        assume(offset <= gap)

        left_arr = np.asarray(left_vals, dtype="float64")
        right_arr = np.asarray(right_vals, dtype="float64")

        records = [
            _make_record(0, left_arr, confidence=left_conf),
            _make_record(gap, right_arr, confidence=right_conf),
        ]

        target_ts = np.asarray([
            _BASE_TS + np.timedelta64(int(offset), "s"),
        ])

        aligned = sd._align_asset_interpolate(
            records=records,
            target_times=target_ts,
            representation="top_of_book",
            missing_policy_large="error",
            large_gap_seconds=gap + 10,
            hybrid_levels=None,
            fail_on_invalid=True,
            asset_name="TEST",
        )

        alpha = offset / float(gap) if gap > 0 else 0.0
        expected = left_arr + alpha * (right_arr - left_arr)
        np.testing.assert_allclose(aligned[0].snapshot_features[:4], expected, rtol=1e-6, atol=1e-6)

        expected_conf = left_conf + alpha * (right_conf - left_conf)
        self.assertAlmostEqual(aligned[0].confidence, expected_conf, places=6)

    def test_align_asset_interpolate_hybrid_linear(self) -> None:
        left_snap = np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]], dtype="float32")
        right_snap = left_snap + 2.0

        records = [
            _make_hybrid_record(0, left_snap, confidence=0.2),
            _make_hybrid_record(10, right_snap, confidence=0.8),
        ]

        target_ts = np.asarray([
            _BASE_TS + np.timedelta64(5, "s"),
        ])

        aligned = sd._align_asset_interpolate(
            records=records,
            target_times=target_ts,
            representation="hybrid",
            missing_policy_large="error",
            large_gap_seconds=60,
            hybrid_levels=left_snap.shape[0],
            fail_on_invalid=True,
            asset_name="TEST",
        )

        expected = left_snap + 0.5 * (right_snap - left_snap)
        np.testing.assert_allclose(aligned[0].hybrid_snapshot, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(aligned[0].snapshot_features[:4], expected[0], rtol=1e-6, atol=1e-6)


class TestSnapshotAlignmentBehavior(unittest.TestCase):
    def test_align_asset_bucket_nearest_within_tolerance(self) -> None:
        records = [
            _make_record(1, np.array([100.0, 1.0, 101.0, 1.0])),
            _make_record(9, np.array([200.0, 1.0, 201.0, 1.0])),
        ]

        target_ts = np.asarray([
            _BASE_TS,
        ])

        aligned = sd._align_asset_bucket(
            records=records,
            target_times=target_ts,
            representation="top_of_book",
            missing_policy_large="error",
            bucket_tolerance_seconds=2.0,
            cadence_seconds=10,
            hybrid_levels=None,
            fail_on_invalid=True,
            asset_name="TEST",
        )

        self.assertEqual(aligned[0].snapshot_features[:4], [100.0, 1.0, 101.0, 1.0])

    def test_align_asset_bucket_zero_pad_when_missing(self) -> None:
        records = [
            _make_record(7, np.array([100.0, 1.0, 101.0, 1.0])),
        ]

        target_ts = np.asarray([
            _BASE_TS,
        ])

        aligned = sd._align_asset_bucket(
            records=records,
            target_times=target_ts,
            representation="top_of_book",
            missing_policy_large="zero_pad",
            bucket_tolerance_seconds=1.0,
            cadence_seconds=10,
            hybrid_levels=None,
            fail_on_invalid=False,
            asset_name="TEST",
        )

        self.assertEqual(aligned[0].snapshot_features[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(aligned[0].confidence, 0.0)

    def test_align_asset_interpolate_large_gap_zero_pad(self) -> None:
        records = [
            _make_record(0, np.array([100.0, 1.0, 101.0, 1.0])),
            _make_record(1000, np.array([200.0, 1.0, 201.0, 1.0])),
        ]

        target_ts = np.asarray([
            _BASE_TS + np.timedelta64(500, "s"),
        ])

        aligned = sd._align_asset_interpolate(
            records=records,
            target_times=target_ts,
            representation="top_of_book",
            missing_policy_large="zero_pad",
            large_gap_seconds=100,
            hybrid_levels=None,
            fail_on_invalid=False,
            asset_name="TEST",
        )

        self.assertEqual(aligned[0].snapshot_features[:4], [0.0, 0.0, 0.0, 0.0])

    def test_align_asset_interpolate_zero_pad_with_fail_on_invalid(self) -> None:
        records = [
            _make_record(10, np.array([100.0, 1.0, 101.0, 1.0])),
        ]

        target_ts = np.asarray([
            _BASE_TS,
        ])

        aligned = sd._align_asset_interpolate(
            records=records,
            target_times=target_ts,
            representation="top_of_book",
            missing_policy_large="zero_pad",
            large_gap_seconds=60,
            hybrid_levels=None,
            fail_on_invalid=True,
            asset_name="TEST",
        )

        self.assertEqual(aligned[0].snapshot_features[:4], [0.0, 0.0, 0.0, 0.0])

    def test_align_asset_records_no_records_error(self) -> None:
        with self.assertRaises(ValueError):
            sd._align_asset_records(
                records=[],
                target_times=np.asarray([_BASE_TS]),
                method="interpolate",
                representation="top_of_book",
                missing_policy_large="error",
                large_gap_seconds=60,
                bucket_tolerance_seconds=0.0,
                cadence_seconds=10,
                hybrid_levels=None,
                fail_on_invalid=True,
                asset_name="TEST",
            )

    def test_align_multi_asset_zero_pad_missing_asset(self) -> None:
        target_records = [
            _make_record(0, np.array([100.0, 1.0, 101.0, 1.0])),
            _make_record(10, np.array([110.0, 1.0, 111.0, 1.0])),
        ]

        asset_records = {
            "BTCUSDT": target_records,
            "ETHUSDT": [],
        }

        alignment_cfg = {
            "method": "interpolate",
            "missing_policy_large": "zero_pad",
            "large_gap_seconds": 60,
            "bucket_tolerance_seconds": 1.0,
        }

        aligned = sd._align_multi_asset_records(
            asset_records=asset_records,
            assets=["BTCUSDT", "ETHUSDT"],
            target_asset="BTCUSDT",
            alignment_cfg=alignment_cfg,
            representation="top_of_book",
            cadence_seconds=10,
            hybrid_levels=None,
            fail_on_invalid=True,
        )

        self.assertEqual(len(aligned), len(target_records))
        for rec in aligned:
            self.assertIn("ETHUSDT", rec.asset_snapshots)
            eth = rec.asset_snapshots["ETHUSDT"]
            self.assertEqual(eth.snapshot_features[:4], [0.0, 0.0, 0.0, 0.0])
            self.assertEqual(eth.confidence, 0.0)

    def test_align_multi_asset_skip_missing_asset(self) -> None:
        target_records = [
            _make_record(0, np.array([100.0, 1.0, 101.0, 1.0])),
            _make_record(10, np.array([110.0, 1.0, 111.0, 1.0])),
        ]

        asset_records = {
            "BTCUSDT": target_records,
            "ETHUSDT": [],
        }

        alignment_cfg = {
            "method": "interpolate",
            "missing_policy_large": "skip",
            "large_gap_seconds": 60,
            "bucket_tolerance_seconds": 1.0,
        }

        aligned = sd._align_multi_asset_records(
            asset_records=asset_records,
            assets=["BTCUSDT", "ETHUSDT"],
            target_asset="BTCUSDT",
            alignment_cfg=alignment_cfg,
            representation="top_of_book",
            cadence_seconds=10,
            hybrid_levels=None,
            fail_on_invalid=False,
        )

        self.assertEqual(len(aligned), 0)

    def test_mask_channels_unchanged_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            x_base = np.asarray(
                [
                    [[[[1.0, 2.0]]]],
                    [[[[2.0, 4.0]]]],
                    [[[[3.0, 6.0]]]],
                    [[[[4.0, 8.0]]]],
                ],
                dtype="float32",
            )
            mask = np.asarray(
                [
                    [[[[0.2, 0.8]]]],
                    [[[[0.2, 0.8]]]],
                    [[[[0.2, 0.8]]]],
                    [[[[0.2, 0.8]]]],
                ],
                dtype="float32",
            )
            x = np.concatenate([x_base, mask], axis=-1)

            y_up = np.zeros((4,), dtype="int64")
            y_down = np.zeros((4,), dtype="int64")
            anchor_ts = np.arange(4, dtype="int64")

            chunk_path = os.path.join(tmp_dir, "chunk.npz")
            np.savez_compressed(chunk_path, x=x, y_up=y_up, y_down=y_down, anchor_ts=anchor_ts)

            chunk = sd.SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=chunk_path,
                num_samples=4,
                start_index=0,
            )
            dataset = sd.SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=4,
                config_hash="hash",
            )

            mask_start = 2
            mask_count = 2
            stats = sd.compute_normalization_stats(
                dataset,
                0,
                4,
                method="min_max",
                mask_start=mask_start,
                mask_count=mask_count,
            )

            x_norm = sd._apply_normalization(x, stats, mask_start=mask_start, mask_count=mask_count)

            np.testing.assert_allclose(x_norm[..., mask_start:mask_start + mask_count], mask, rtol=0, atol=0)

            x_flat = x_base.reshape(x_base.shape[0], -1)
            expected_min = np.min(x_flat, axis=0)
            expected_max = np.max(x_flat, axis=0)
            np.testing.assert_allclose(stats.min, expected_min)
            np.testing.assert_allclose(stats.max, expected_max)

    def test_gap_reset_prevents_cross_gap_samples(self) -> None:
        config = {
            "data": {
                "time_range": {"cadence_seconds": 10},
                "order_book": {"representation": "top_of_book"},
                "asset_pairs": {
                    "target_asset": "BTCUSDT",
                    "correlated_assets": [],
                    "alignment": {"include_mask_channel": False},
                },
                "temporal_features": {"local": [], "global": [], "market_session": {}},
            },
            "targets": {
                "visible_window_seconds": 30,
                "prediction_horizon_seconds": 10,
                "price_classes": {"boundaries": [0.1, 0.2]},
            },
            "model": {
                "output": {"type": "two_head_intensity", "num_classes": 3},
                "input_representation": {
                    "temporal_features": {
                        "integration_mode": "none",
                        "use_local_features": False,
                        "use_global_features": False,
                    }
                },
            },
            "preprocessing": {"feature_engineering": {"enabled": False}},
        }

        builder = sd.StreamingSampleBuilder(
            config=config,
            representation="top_of_book",
            height=2,
            width=2,
            assets=["BTCUSDT"],
            target_asset="BTCUSDT",
        )

        def _make_multi(offset_s: int, gap_reset: bool = False) -> sd.MultiAssetSnapshotRecord:
            rec = _make_record(offset_s, np.array([100.0, 1.0, 101.0, 1.0]), gap_reset=gap_reset)
            return sd.MultiAssetSnapshotRecord(timestamp=rec.timestamp, asset_snapshots={"BTCUSDT": rec})

        snapshots = [
            _make_multi(0),
            _make_multi(10),
            _make_multi(20),
            _make_multi(30),
            _make_multi(40),
            _make_multi(50, gap_reset=True),
            _make_multi(60),
            _make_multi(70),
            _make_multi(80),
        ]

        samples = []
        for snapshot in snapshots:
            samples.extend(builder.add_snapshot(snapshot))

        base_seconds = int(_BASE_TS.astype("datetime64[s]").astype("int64"))
        anchor_offsets = [s.anchor_ts_seconds - base_seconds for s in samples]
        self.assertEqual(anchor_offsets, [20, 30, 70])


class TestInterpolationPrecisionDiagnostics(unittest.TestCase):
    def test_interpolation_precision_vs_gap_size(self) -> None:
        base_times = np.arange(0, 2001, 10)
        gap_errors: dict[int, list[float]] = {}

        for seed in (1, 2, 3, 4):
            rng = np.random.default_rng(seed)
            keep_mask = rng.random(len(base_times)) > 0.2
            keep_mask[0] = True
            keep_mask[-1] = True
            sample_times = base_times[keep_mask]
            if sample_times.size < 3:
                continue

            records = [
                _make_record(int(t), _synthetic_quadratic_features(float(t)))
                for t in sample_times
            ]

            mid_times = []
            gap_sizes = []
            for left, right in zip(sample_times[:-1], sample_times[1:]):
                gap = int(right - left)
                if gap <= 0:
                    continue
                mid_times.append(int(left + gap / 2))
                gap_sizes.append(gap)

            if not mid_times:
                continue

            target_ts = np.asarray([
                _BASE_TS + np.timedelta64(t, "s") for t in mid_times
            ])

            aligned = sd._align_asset_interpolate(
                records=records,
                target_times=target_ts,
                representation="top_of_book",
                missing_policy_large="error",
                large_gap_seconds=3600,
                hybrid_levels=None,
                fail_on_invalid=True,
                asset_name="TEST",
            )

            for rec, mid, gap in zip(aligned, mid_times, gap_sizes):
                if rec is None:
                    continue
                predicted = np.asarray(rec.snapshot_features[:4], dtype="float64")
                true_val = _synthetic_quadratic_features(float(mid))
                error = float(np.mean(np.abs(predicted - true_val)))
                gap_errors.setdefault(gap, []).append(error)

        candidate_gaps = [gap for gap, errs in gap_errors.items() if len(errs) >= 5]
        if len(candidate_gaps) < 2:
            self.skipTest("Insufficient gap sizes for interpolation precision diagnostics")

        candidate_gaps.sort()
        smallest = candidate_gaps[0]
        largest = candidate_gaps[-1]

        errors_small = np.asarray(gap_errors[smallest], dtype="float64")
        errors_large = np.asarray(gap_errors[largest], dtype="float64")

        rng = np.random.default_rng(12345)
        diffs = []
        for _ in range(200):
            sample_small = rng.choice(errors_small, size=errors_small.size, replace=True)
            sample_large = rng.choice(errors_large, size=errors_large.size, replace=True)
            diffs.append(float(np.mean(sample_large) - np.mean(sample_small)))

        lower_bound = float(np.percentile(diffs, 5))
        self.assertGreater(lower_bound, 0.0)

        means = [float(np.mean(gap_errors[gap])) for gap in candidate_gaps]
        for prev, curr in zip(means, means[1:]):
            self.assertGreaterEqual(curr + 1e-6, prev)

        plot_dir = os.environ.get("SNAPSHOT_ALIGNMENT_PLOT_DIR")
        if plot_dir:
            try:
                import matplotlib.pyplot as plt  # type: ignore[import]
            except Exception:
                return

            os.makedirs(plot_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(candidate_gaps, means, marker="o")
            ax.set_xlabel("Gap size (seconds)")
            ax.set_ylabel("Mean absolute error")
            ax.set_title("Interpolation error vs gap size")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "interpolation_error_vs_gap.png"))
            plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
