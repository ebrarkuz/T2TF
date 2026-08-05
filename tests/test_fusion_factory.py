import unittest

import numpy as np

from fusion import available_algorithms, create_fusion_algorithm


def measurement():
    return {
        "timestamp": 1.0,
        "state": np.array([1000.0, 50.0, 500.0, 10.0, 3000.0, 2.0]).reshape(6, 1),
        "cov": np.eye(6) * 25.0,
        "tq": 12.0,
        "src": ("RADAR_A", "A-1"),
        "measurement_id": "m-1",
        "sequence_number": 1,
        "measurement_timestamp": 1.0,
    }


class FusionFactoryTests(unittest.TestCase):
    def test_all_advertised_algorithms_process_a_measurement(self):
        self.assertEqual(set(available_algorithms()), {"basic_cv", "advanced_ca", "dual_imm"})
        for name in available_algorithms():
            with self.subTest(name=name):
                algorithm = create_fusion_algorithm(name, {})
                tracks = algorithm.process_measurement(measurement())
                self.assertEqual(len(tracks), 1)
                self.assertTrue(tracks[0]["measurement_used"])
                self.assertEqual(algorithm.get_diagnostics()["algorithm"], name)
                algorithm.reset()
                self.assertEqual(algorithm.get_diagnostics()["active_track_count"], 0)

    def test_unknown_algorithm_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Bilinmeyen.*basic_cv.*dual_imm"):
            create_fusion_algorithm("not-real", {})

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Desteklenmeyen parametre"):
            create_fusion_algorithm("dual_imm", {"invented_gate": 1})


if __name__ == "__main__":
    unittest.main()
