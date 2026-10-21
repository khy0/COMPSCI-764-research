import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import stats_analysis as sa


class T(unittest.TestCase):
    def test_holm_matches_hand(self):
        np.testing.assert_allclose(sa.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])

    def test_crossover_interp_and_censoring(self):
        ns = np.array([70., 175., 350., 700.])
        n, st = sa.crossover_n(np.array([.5, .7, .9, .95]), ns, 0.8, 1)
        self.assertEqual(st, "interp"); self.assertTrue(175 < n < 350)
        self.assertEqual(sa.crossover_n(np.array([.9, .9, .9, .9]), ns, 0.8, 1)[1], "below_min")
        self.assertEqual(sa.crossover_n(np.array([.5, .5, .5, .5]), ns, 0.8, 1)[1], "never")
        # lower-is-better metric
        self.assertEqual(sa.crossover_n(np.array([30, 20, 10, 5.]), ns, 12, -1)[1], "interp")
        # exactly halfway in log space -> geometric mean
        n, _ = sa.crossover_n(np.array([0., 1.]), np.array([100., 400.]), 0.5, 1)
        self.assertAlmostEqual(n, 200.0)

    def test_bootstrap_ci_covers_mean_and_two_stage_wider(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.8, 0.1, size=(200, 1)); u = a + rng.normal(0, 0.05, (200, 3)) + np.array([0, .05, -.05])
        d1 = sa.bootstrap_conditions({"u": u}, "mean", 3000, np.random.default_rng(1), False)["u"]
        d2 = sa.bootstrap_conditions({"u": u}, "mean", 3000, np.random.default_rng(1), True)["u"]
        lo, hi = sa.ci(d1, .95)
        self.assertTrue(lo < u.mean() < hi)
        self.assertGreater(np.ptp(sa.ci(d2, .95)), np.ptp(sa.ci(d1, .95)))

    def test_run_id_parse_and_duplicate_guard(self):
        import pandas as pd, tempfile
        cfg = {"metrics": {"dice": {}}}
        rows = [dict(arm="unet", run_id="10pct_seed2", image_id="a", dice=.5)] * 2
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); pd.DataFrame(rows).to_csv(f.name, index=False)
        with self.assertRaises(ValueError): sa.load_results(f.name, cfg)
        pd.DataFrame(rows[:1]).to_csv(f.name, index=False)
        d = sa.load_results(f.name, cfg); self.assertEqual((d.bracket[0], int(d.seed[0])), ("10pct", 2))

if __name__ == "__main__":
    unittest.main()
