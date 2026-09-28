from .losses import LinearLoss
from .linear_model import LinearModel

import numpy as np
from scipy.sparse import csr_matrix, hstack
import numpy.testing as npt
import time
import pytest


def test_function():
    loss_function = LinearLoss(l2_coef=1.0)
    X = np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ])
    y = np.array([-1, 1, 1])
    w = np.array([1, 2, 3])
    npt.assert_almost_equal(loss_function.func(X, y, w), 175.66666666666669, decimal=3)


def test_function_negative_inf_values():
    loss_function = LinearLoss(l2_coef=0.0)
    X = np.array([
        [1, 10 ** 5],
        [1, -10 ** 5],
        [1, 10 ** 5]
    ])
    y = np.array([1, -1, 1])
    w = np.array([1, 100])
    npt.assert_almost_equal(loss_function.func(X, y, w), 99999986666667.97, decimal=1)


def test_function_positive_inf_values():
    loss_function = LinearLoss(l2_coef=0.0)
    X = np.array([
        [1, 10 ** 2],
        [1, -10 ** 2],
        [1, 10 ** 2]
    ])
    y = np.array([-1, 1, -1])
    w = np.array([1, 100])
    npt.assert_almost_equal(loss_function.func(X, y, w), 100026669.33333334, decimal=3)


def test_gradient():
    loss_function = LinearLoss(l2_coef=1.0)
    X = np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ])
    y = np.array([-1, 1, 1])
    w = np.array([1, 2, 3])
    right_gradient = np.array([24.0, 20., 99.33333333])
    npt.assert_almost_equal(loss_function.grad(X, y, w), right_gradient, decimal=3)


def test_function_sparse():
    loss_function = LinearLoss(l2_coef=1.0)
    X = csr_matrix(np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ]))
    y = np.array([-1, 1, 1])
    w = np.array([1, 2, 3])
    npt.assert_almost_equal(loss_function.func(X, y, w),  175.66666666666669, decimal=3)


def test_gradient_sparse():
    loss_function = LinearLoss(l2_coef=1.0)
    X = csr_matrix(np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ]))
    y = np.array([-1, 1, 1])
    w = np.array([1, 2, 3])
    right_gradient = np. array([24.0, 20., 99.33333333])
    npt.assert_almost_equal(loss_function.grad(X, y, w), right_gradient, decimal=5)


def create_simple_dataset():
    X = (np.random.randint(1, 4, (1000, 20)))
    w = np.random.randint(0, 5, 20)
    y = X @ w
    return X, y


def test_simple_classification_task():
    X, y = create_simple_dataset()
    loss_function = LinearLoss(l2_coef=0.0)
    linear_model = LinearModel(
        loss_function=loss_function,
        batch_size=100,
        step_alpha=0.0001,
        step_beta=0,
        tolerance=1e-6,
        max_iter=10000,
    )
    linear_model.fit(X, y)
    predictions = linear_model.predict(X)
    npt.assert_almost_equal(y, predictions, decimal=2)


def test_logging():
    X, y = create_simple_dataset()
    loss_function = LinearLoss(l2_coef=0.1)
    linear_model = LinearModel(
        loss_function=loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0,
        tolerance=1e-100,
        max_iter=5,
    )
    history = linear_model.fit(X, y, trace=True)
    for key in ['time', 'func']:
        assert key in history
        assert len(history[key]) == 5


@pytest.mark.parametrize("step_alpha, step_beta, answer", [
    (5e-3, 0.80, 0.367),
    (1e-2, 0.99, 0.365),
    (2e-1, 2.45, 0.375),
])
def test_full_gd(step_alpha, step_beta, answer):
    X = csr_matrix(np.array([
        [1, 0, 0, 2, 5, 0.9],
        [1, 5, 1, 3, 1, 0.1],
        [1, 0, 0, 2, 1, 0.5],
        [1, 5, 1, 4, 3, 0.32],
        [1, 0, 2, 3, 2, 0.1],
        [1, 5, 2, 5, 4, 0.10],
        [1, 0, 0, 6, 6, 0.28],
        [1, 5, 1, 3, 2, 0.7],
    ]))

    y = np.array([-0.57664669, -1.55845798, -0.86967977, -1.69210847,
                  -1.18721522, -1.92368875, -1.31763832, -1.67671109])
    w_0 = np.array([-0.2, -0.08, -0.15, -0.2, 0.1, 0.3])

    loss_function = LinearLoss(l2_coef=5)
    lm = LinearModel(
        loss_function=loss_function,
        step_alpha=step_alpha,
        step_beta=step_beta,
        tolerance=1e-5,
        max_iter=1000,
    )
    lm.fit(X, y, w_0=w_0)
    npt.assert_almost_equal(lm.loss_function.func(X, y, lm.get_weights()), answer, decimal=3)


def test_real_sparse_problem():
    data = np.array([1, 1, 1, 1, 1])
    row_ind = np.array([0, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
    col_ind = np.array([0, 10 ** 4, 10 ** 5, 10 ** 5, 10 ** 6])
    X = csr_matrix((data, (row_ind, col_ind)))
    X = csr_matrix(hstack([csr_matrix(np.ones((X.shape[0], 1))), X]))
    y = np.array([1] * (10 ** 6 + 1))
    y[:5 * 10 ** 5] = -1
    w = np.ones(10 ** 6 + 2)
    loss_function = LinearLoss(l2_coef=5)
    start = time.time()
    loss_function.func(X, y, w)
    finish = time.time() - start
    assert finish < 1
