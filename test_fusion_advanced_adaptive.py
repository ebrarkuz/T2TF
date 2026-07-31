import math
import unittest

import numpy as np

import fusion_advanced as advanced


class AdvancedAdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.state = np.zeros((6, 1))
        self.covariance = np.eye(6)

    def make_track(self, src=("R1", "L1")):
        track = advanced.GlobalTrack(0.0, self.state, self.covariance, 15, src)
        track.cov = np.eye(9)
        return track

    def measurement_with_nis(self, nis, src):
        # With P=I and R=I at t=0, x-offset sqrt(2*NIS) produces the requested NIS.
        state = self.state.copy()
        state[0, 0] = math.sqrt(2.0 * nis)
        return {"state": state, "cov": self.covariance, "tq": 15, "src": src}

    def test_track_starts_in_normal_mode(self):
        track = self.make_track()
        self.assertEqual(track.q_scale, 1.0)
        self.assertEqual(track.last_nis, 0.0)
        self.assertEqual(track.maneuver_score, 0.0)
        self.assertEqual(track.get_effective_gate(), advanced.BASE_GATE_CHI2_6DOF)

    def test_repeated_high_accepted_nis_increases_q_and_gate(self):
        track = self.make_track()
        for _ in range(8):
            track.update_maneuver_adaptation(20.0)
        self.assertGreater(track.q_scale, 1.0)
        self.assertGreater(track.get_effective_gate(), advanced.BASE_GATE_CHI2_6DOF)
        self.assertLessEqual(
            track.get_effective_gate(),
            advanced.BASE_GATE_CHI2_6DOF * advanced.GATE_SCALE_MAX,
        )

    def test_single_high_nis_does_not_expand_gate(self):
        track = self.make_track()
        track.update_maneuver_adaptation(20.0)
        self.assertLessEqual(track.maneuver_score, 0.6)
        self.assertEqual(track.get_effective_gate(), advanced.BASE_GATE_CHI2_6DOF)

    def test_rejected_source_map_measurement_does_not_adapt(self):
        center = advanced.FusionCenter()
        track = self.make_track()
        center.tracks = [track]
        center.src_map[("R1", "L1")] = track.id
        center.process_batch(0.0, [self.measurement_with_nis(40.0, ("R1", "L1"))])
        self.assertEqual(track.last_nis, 0.0)
        self.assertEqual(track.q_scale, 1.0)

    def test_adaptive_gate_is_used_for_source_map_association(self):
        center = advanced.FusionCenter()
        track = self.make_track()
        track.q_scale = 4.0
        track.maneuver_score = 1.0  # effective gate = 40.5, so NIS=40 is accepted
        center.tracks = [track]
        center.src_map[("R1", "L1")] = track.id
        center.process_batch(0.0, [self.measurement_with_nis(40.0, ("R1", "L1"))])
        self.assertAlmostEqual(track.last_nis, 40.0, places=8)

    def test_adaptive_gate_is_used_for_global_assignment(self):
        center = advanced.FusionCenter()
        track = self.make_track()
        track.q_scale = 4.0
        track.maneuver_score = 1.0
        center.tracks = [track]
        center.process_batch(0.0, [self.measurement_with_nis(40.0, ("R2", "L2"))])
        self.assertAlmostEqual(track.last_nis, 40.0, places=8)
        self.assertEqual(center.src_map[("R2", "L2")], track.id)

    def high_velocity_measurement(self, src):
        state = self.state.copy()
        state[1, 0] = advanced.MAX_ASSOC_VEL_DIFF_MPS + 1.0
        covariance = self.covariance.copy()
        covariance[1, 1] = 1e6  # Keep NIS inside the gate; velocity residual must reject it.
        return {"state": state, "cov": covariance, "tq": 15, "src": src}

    def test_velocity_residual_rejects_source_map_association(self):
        center = advanced.FusionCenter()
        track = self.make_track()
        center.tracks = [track]
        center.src_map[("R1", "L1")] = track.id
        center.process_batch(0.0, [self.high_velocity_measurement(("R1", "L1"))])
        self.assertEqual(track.last_nis, 0.0)

    def test_velocity_residual_rejects_global_assignment(self):
        center = advanced.FusionCenter()
        track = self.make_track()
        center.tracks = [track]
        center.process_batch(0.0, [self.high_velocity_measurement(("R2", "L2"))])
        self.assertEqual(track.last_nis, 0.0)
        self.assertNotEqual(center.src_map[("R2", "L2")], track.id)

    def test_white_jerk_q_and_xy_maneuver_gain(self):
        track = self.make_track()
        track.cov = np.zeros((9, 9))
        track.propagate(2.0)
        q = advanced.PROCESS_NOISE_INTENSITY ** 2
        self.assertAlmostEqual(track.cov[0, 0], q * (2.0**5 / 20.0))
        self.assertAlmostEqual(track.cov[2, 2], q * 2.0)

        maneuver_track = self.make_track()
        maneuver_track.cov = np.zeros((9, 9))
        maneuver_track.q_scale = 4.0
        maneuver_track.propagate(1.0)
        np.testing.assert_allclose(
            maneuver_track.cov[0:3, 0:3],
            advanced.XY_MANEUVER_Q_GAIN * maneuver_track.cov[6:9, 6:9],
        )


if __name__ == "__main__":
    unittest.main()
