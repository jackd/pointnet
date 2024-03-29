#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
from pointnet.augment.util.interp import linear_interp
from tensorflow_datasets.testing.test_utils import run_in_graph_and_eager_modes


class TestInterp(tf.test.TestCase):

    @run_in_graph_and_eager_modes()
    def test_intercepts3d(self):
        grid = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]],
                        dtype=np.float32)
        grid = np.stack([grid, grid + 100, grid + 200, grid + 300])
        coords = np.array([
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=np.float32)
        tf_vals = self.evaluate(linear_interp(grid, coords))
        expected = np.array([111, 11])
        np.testing.assert_allclose(tf_vals, expected)
        shift = np.random.randn(*coords.shape) * 1e-5
        max_shift = 1e-5
        too_big = np.abs(shift) > max_shift
        shift[too_big] *= max_shift / shift[too_big]
        coords += shift
        close_vals = self.evaluate(linear_interp(grid, coords))
        np.testing.assert_allclose(close_vals, expected, atol=1e-2)

    def test_known_vals1d(self):
        grid = [10, 11, 14]
        coords = [[0], [0.5], [1.2]]
        expected = np.array([10, 10.5, 11.6])
        actual = self.evaluate(linear_interp(grid, coords))
        np.testing.assert_allclose(actual, expected)

    def test_intercepts3d_batch(self):
        grid = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]],
                        dtype=np.float32)
        grid = np.stack([grid, grid + 100, grid + 200, grid + 300])
        coords = np.array([
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=np.float32)
        grid = np.stack([grid, grid + 1000])
        coords = np.stack([coords, coords])
        actual = self.evaluate(linear_interp(grid, coords))
        expected = np.array([111, 11])
        expected = np.stack([expected, expected + 1000])
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # tf.logging.set_verbosity(0)
    tf.compat.v1.enable_eager_execution()
    unittest.main()
