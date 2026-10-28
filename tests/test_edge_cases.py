"""
Integration tests for Edge & Failure Case Suite (Superposition, Dropout, Velocity Jump, Baseline Drift).
"""

import unittest
from src.edge_case_runner import EdgeCaseSimulator

class TestEdgeCases(unittest.TestCase):
    def test_all_four_edge_cases(self):
        sim = EdgeCaseSimulator()
        
        c1 = sim.run_case1_superposition()
        c2 = sim.run_case2_sensor_dropout()
        c3 = sim.run_case3_velocity_shift()
        c4 = sim.run_case4_sensor_drift()
        
        self.assertTrue(c1["physics_success"])
        self.assertTrue(c2["physics_success"])
        self.assertTrue(c3["adaptive_physics_success"])
        self.assertTrue(c4["ewma_detected"])
        self.assertGreater(c4["static_false_positives"], c4["ewma_false_positives"])

if __name__ == "__main__":
    unittest.main()
