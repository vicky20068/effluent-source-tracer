"""
Unit tests for synthetic and empirical dataset generation modules.
"""

import unittest
import pandas as pd
from src.data_generator import generate_synthetic_dataset, is_unit_operating
from datetime import datetime

class TestDataGenerator(unittest.TestCase):
    def test_synthetic_dataset_dimensions(self):
        df, gt = generate_synthetic_dataset(days=2, freq_minutes=5, num_anomalies=3, random_seed=42)
        expected_rows = 2 * 24 * 12  # 576 steps
        self.assertEqual(len(df), expected_rows)
        self.assertEqual(len(gt), 3)
        self.assertIn("COD_mgL", df.columns)
        self.assertIn("HeavyMetals_mgL", df.columns)
        self.assertIn("pH", df.columns)
        self.assertIn("TDS_mgL", df.columns)
        self.assertIn("FlowRate_m3h", df.columns)

    def test_unit_operating_shifts(self):
        # U1 operates 08:00 - 16:00
        t_active = datetime(2026, 9, 10, 10, 30)
        t_inactive = datetime(2026, 9, 10, 20, 0)
        
        self.assertTrue(is_unit_operating("U1", t_active))
        self.assertFalse(is_unit_operating("U1", t_inactive))

if __name__ == "__main__":
    unittest.main()
