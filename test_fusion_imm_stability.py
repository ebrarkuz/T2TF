import math
import unittest

import numpy as np

import fusion_imm as imm


class IMMStabilityTests(unittest.TestCase):
    def setUp(self):
        self.measurement_cov = np.diag([25.0, 0.25, 25.0, 0.25, 25.0, 0.25])

    @staticmethod
    def cv_state(t):
        return np.array([100*t, 100, 20*t, 20, 1000, 0.0]).reshape(6, 1)

    @staticmethod
    def ca_state(t):
        return np.array([100*t + 2*t*t, 100 + 4*t, 20*t, 20, 1000, 0.0]).reshape(6, 1)

    @staticmethod
    def ct_left_state(t):
        speed = 200.0
        omega = imm.OMEGA_LEFT
        return np.array([
            speed * math.sin(omega*t) / omega,
            speed * math.cos(omega*t),
            speed * (1 - math.cos(omega*t)) / omega,
            speed * math.sin(omega*t),
            1000,
            0.0,
        ]).reshape(6, 1)

    def run_motion(self, state_function, steps=30):
        track = imm.GlobalTrackIMM3(
            0.0, state_function(0.0), self.measurement_cov, 15, ("R1", "L1")
        )
        for step in range(1, steps + 1):
            track.propagate(float(step))
            track.update(
                state_function(float(step)), self.measurement_cov, 15, ("R1", "L1")
            )
        return track

    def test_linear_target_prefers_cv(self):
        track = self.run_motion(self.cv_state)
        self.assertEqual(max(track.xy_mode_prob, key=track.xy_mode_prob.get), "CV")

    def test_constant_acceleration_raises_ca(self):
        track = self.run_motion(self.ca_state)
        self.assertEqual(max(track.xy_mode_prob, key=track.xy_mode_prob.get), "CA")

    def test_constant_turn_prefers_ct(self):
        track = self.run_motion(self.ct_left_state)
        self.assertEqual(max(track.xy_mode_prob, key=track.xy_mode_prob.get), "CT_LEFT")

    def test_measurement_gap_coasts_then_decays(self):
        track = imm.GlobalTrackIMM3(
            0.0, self.cv_state(0.0), self.measurement_cov, 15, ("R1", "L1")
        )
        for step in range(1, 5):
            radar = "R1" if step % 2 else "R2"
            track.propagate(float(step))
            track.update(
                self.cv_state(float(step)), self.measurement_cov, 15, (radar, "L1")
            )
        self.assertEqual(track.status, "CONFIRMED")
        probability_before_gap = track.existence_prob
        track.propagate(7.0)
        self.assertNotEqual(track.status, "DELETED")
        self.assertLess(track.existence_prob, probability_before_gap)

    def test_two_clutter_hits_do_not_confirm(self):
        track = imm.GlobalTrackIMM3(
            0.0, self.cv_state(0.0), self.measurement_cov, 15, ("R1", "CL1")
        )
        track.propagate(1.0)
        track.update(self.cv_state(1.0), self.measurement_cov, 15, ("R2", "CL2"))
        self.assertEqual(track.hits_count, 2)
        self.assertEqual(track.status, "TENTATIVE")

    def test_covariances_remain_psd(self):
        track = self.run_motion(self.ca_state, steps=12)
        matrices = list(track.xy_model_cov.values()) + list(track.z_model_cov.values())
        matrices.append(track.cov)
        for covariance in matrices:
            self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-9)

    def test_mode_probabilities_sum_to_one(self):
        track = self.run_motion(self.ct_left_state, steps=12)
        self.assertAlmostEqual(sum(track.xy_mode_prob.values()), 1.0, places=12)
        self.assertAlmostEqual(sum(track.z_mode_prob.values()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
