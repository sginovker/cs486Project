import math
from unittest import TestCase

from src.d3 import euclidean_distance


class TestEuclideanDistance(TestCase):
    def test_euclidean_distance_empty(self):
        self.assertAlmostEqual(
            euclidean_distance([], []),
            0.0,
            places=8,
        )

    def test_euclidean_distance_single(self):
        self.assertAlmostEqual(
            euclidean_distance([1], [3]),
            2.0,
            places=8,
        )

    def test_euclidean_distance_multiple(self):
        self.assertAlmostEqual(
            euclidean_distance([1, 2, 3], [3, 2, 1]),
            math.sqrt(8),
            places=8,
        )
