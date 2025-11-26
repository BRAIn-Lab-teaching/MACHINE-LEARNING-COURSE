from unittest import TestCase, mock

import numpy as np

from ensemble.sampler import BaseSampler, FeatureSampler, ObjectSampler


class BaseSamplerTests(TestCase):
    def test_number_of_samples(self):
        sampler = BaseSampler(max_samples=0.2)

        for num_objects in range(2, 11):
            num_sampled = len(sampler.sample_indices(num_objects))
            self.assertEqual(num_sampled, int(sampler.max_samples * num_objects))

        sampler = BaseSampler(max_samples=0.7)

        for num_objects in range(2, 11):
            num_sampled = len(sampler.sample_indices(num_objects))
            self.assertEqual(num_sampled, int(sampler.max_samples * num_objects))

    def test_bootstrap_mode(self):
        sampler = BaseSampler(max_samples=1.2, bootstrap=True)
        sampled = sampler.sample_indices(10)
        self.assertGreater(len(sampled), len(set(sampled)))

    def test_random_state_compatibility(self):
        # use np.random.choice
        sampler = BaseSampler(max_samples=0.8, random_state=42)
        sampled = list(sampler.sample_indices(10))
        expected = [8, 1, 5, 0, 7, 2, 9, 4]
        self.assertEqual(sampled, expected)


class ObjectsSamplerTests(TestCase):
    @mock.patch('ensemble.sampler.BaseSampler.sample_indices',
                return_value=[1, 3, 0, 4, 3])
    def test_base_scenario(self, indices):
        x = np.arange(18).reshape(6, -1)
        y = np.arange(6)

        sampler = ObjectSampler()
        x_sampled, y_sampled = sampler.sample(x, y)

        indices = indices.return_value
        self.assertTrue(np.all(x_sampled == x[indices]))
        self.assertTrue(np.all(y_sampled == y[indices]))


class FeaturesSamplerTests(TestCase):
    @mock.patch('ensemble.sampler.BaseSampler.sample_indices',
                return_value=[1, 3, 0, 4, 3])
    def test_base_scenario(self, indices):
        x = np.arange(18).reshape(3, -1)

        sampler = FeatureSampler()
        x_sampled = sampler.sample(x)

        indices = indices.return_value
        self.assertTrue(np.all(x_sampled == x[:, indices]))
