from unittest import TestCase

import numpy as np

from itertools import chain, cycle

from ensemble.sampler import BaseSampler
from ensemble.booster import Booster, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


class RandomRegressor:
    def __init__(self, random_state=None):
        self.random_state = np.random.RandomState(random_state)
        self.X = None

    def fit(self, X, y):
        self.X = X
        return self

    def predict(self, X):
        if self.X is None:
            raise RuntimeError('Regressor is not fitted')
        return self.random_state.random(size=(len(X), 1))


class PredefinedSampler(BaseSampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices
        self.indices_yield = cycle(self.indices)

    def sample_indices(self, n_objects):
        return next(self.indices_yield)

    def sample(self, X, y=None):
        indices = self.sample_indices(len(X))
        if y is None:
            return X[indices]
        return X[indices], y[indices]


class GradientBoosstingClassifierTests(TestCase):
    def test_common_scenario(self):
        booster = GradientBoostingClassifier(n_estimators=10)

        X = np.arange(24).reshape(6, 4)
        y = np.random.choice(2, size=len(X), replace=True)

        booster = booster.fit(X, y)
        self.assertEqual(booster.predict(X).shape, (len(X), ))

    def test_fitted_subsamples(self):
        feature_sampler = PredefinedSampler([
            [2, 3, 3],
            [0, 1, 2],
            [3, 2, 1],
            [3, 0, 1],
        ])

        n_estimators = len(feature_sampler.indices)

        booster = GradientBoostingClassifier(n_estimators=n_estimators)
        booster.base_estimator = RandomRegressor
        booster.feature_sampler = feature_sampler
        booster.params = {}

        n_objects = len(feature_sampler.indices)
        n_features = max(chain.from_iterable(feature_sampler.indices), default=-1) + 1

        X = np.arange(n_objects * n_features).reshape(n_objects, n_features)
        y = np.arange(len(X))

        booster = booster.fit(X, y)

        self.assertEqual(len(booster.estimators), n_estimators)
        self.assertEqual(len(booster.indices), n_estimators)

        it_checker = zip(
            booster.estimators,
            booster.indices,
            feature_sampler.indices,
        )

        for estim, indices, features_index in it_checker:
            features_index = np.asarray(features_index)

            self.assertTrue(np.all(indices == features_index))
            self.assertTrue(np.all(estim.X == X[:, features_index]))

    def test_check_predicts(self):
        np.random.seed(56)

        n_batch = 25

        X_train = np.vstack([
            np.random.random(size=(n_batch, 2)) + np.array([0, 0]),
            np.random.random(size=(n_batch, 2)) + np.array([0, 1]),
            np.random.random(size=(n_batch, 2)) + np.array([1, 0]),
            np.random.random(size=(n_batch, 2)) + np.array([1, 1]),
        ])

        y_train = np.r_[np.ones(n_batch), np.zeros(n_batch), np.zeros(n_batch), np.ones(n_batch)]

        booster = GradientBoostingClassifier(n_estimators=20, lr=0.1, max_depth=4, random_state=42,
                                             max_features_samples=1.0)
        booster.fit(X_train, y_train)

        X_valid = np.vstack([
            np.random.random(size=(n_batch, 2)) + np.array([0, 0]),
            np.random.random(size=(n_batch, 2)) + np.array([0, 1]),
            np.random.random(size=(n_batch, 2)) + np.array([1, 0]),
            np.random.random(size=(n_batch, 2)) + np.array([1, 1]),
        ])

        y_predict = booster.predict_proba(X_valid)
        self.assertTrue(np.all((0 <= y_predict) & (y_predict <= 1)))

        y_predict = booster.predict(X_valid)
        self.assertGreaterEqual(accuracy_score(y_predict, y_train), 0.95)
