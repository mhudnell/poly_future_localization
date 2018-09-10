import unittest
import gan_1obj
import numpy as np

class TestData(unittest.TestCase):
  def setUp(self):
    self.past_all, self.future_all, self.past_files, self.future_files = gan_1obj.get_data()

  def test_num_samples_equal(self):
    self.assertEqual(len(self.past_all), len(self.future_all))

  def test_data_past_normalized(self):
    self.assertTrue(np.all(np.logical_and(self.past_all >= 0, self.past_all <= 1)))

  def test_data_future_normalized(self):
    self.assertTrue(np.all(np.logical_and(self.future_all >= 0, self.future_all <= 1)))


if __name__ == '__main__':
    unittest.main()