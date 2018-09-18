import unittest
import gan_1obj
import numpy as np
import tensorflow as tf

class TestData(unittest.TestCase):
    def setUp(self):
        self.past_all, self.future_all, self.past_files, self.future_files = gan_1obj.get_data()

    def test_num_samples_equal(self):
        self.assertEqual(len(self.past_all), len(self.future_all))

    def test_data_past_normalized(self):
        self.assertTrue(np.all(np.logical_and(self.past_all >= 0, self.past_all <= 1)))

    def test_data_future_normalized(self):
        self.assertTrue(np.all(np.logical_and(self.future_all >= 0, self.future_all <= 1)))

class TestLoss(unittest.TestCase):
    
    def test_smoothL1_case1(self):
        g = np.array([0.14352065, 0.43352666, 0.1219445, 0.33704963])
        t = np.array([0.22506863, 0.42474266, 0.11288854, 0.28878626])

        with tf.Session() as sess:
            loss = sess.run(gan_1obj.smoothL1(g, t))

        self.assertEqual(loss, 0.0045692974966794484)  # 0.00456929749668

if __name__ == '__main__':
    unittest.main()
