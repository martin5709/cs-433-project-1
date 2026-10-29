# Code for Step 2

from wk2_gd import gradient_descent
from wk2_sgd import stochastic_gradient_descent
from wk3_copy import least_squares as lsq
from wk3_copy import ridge_regression as rr
from wk5_logistic import logistic_regression as lr
from wk5_logistic import penalized_logistic_regression as plr
from wk5_logistic import sigmoid
from wk5_logistic import f1_logistic
from custom_functions import penalized_logistic_regression_weighted as plrw
import numpy as np

# This is the same as the function implemented during Week 2.
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma)

# This is the same as the function implemented during Week 2.
# Batch Size parameter added for full compatability.
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    return stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)

# This is the same as one of the functions implemented during Week 3.
def least_squares(y, tx):
    return lsq(y, tx)

# This is the same as one of the functions implemented during Week 3.
def ridge_regression(y, tx, lambda_):
    return rr(y, tx, lambda_)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # MSE based. I will return all losses, as anything else does not make sense.
    w = initial_w
    losses = []
    for _ in range(max_iters):
        grad, loss = lr(y, tx, w)
        w -= gamma*grad
        losses.append(loss)
    return losses, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    losses = []
    for _ in range(max_iters):
        grad, loss = plr(y, tx, w, lambda_)
        w -= gamma*grad
        losses.append(loss)
    return losses, w

# Personal Extras

def lr_inference(tx, weights):
    """Inference for logistic regression. Returns the classification. May return list of type (N, 1)"""
    return np.round(sigmoid(tx @ weights))

def reg_logistic_regression_weighted(y, tx, lambda_, initial_w, gamma, dataset_bias):
    """Regularized Logistic Regression, but with the learning objective changed to account for dataset bias and F1 bias. dataset_bias should be the fraction of points that satisfy y == 1"""
    bias = dataset_bias/2.0
    w = initial_w
    losses = []
    grad, loss = plrw(y, tx, w, lambda_, bias)
    w -= gamma*grad
    losses.append(loss)
    return losses, w

def reg_f1_logistic(y, tx, lambda_, initial_w, gamma):
    """Optimizing based on the f1 score."""
    w = initial_w
    losses = []
    grad, loss = f1_logistic(y, tx, w, lambda_)
    w += gamma*grad # Yes, the + is intentional, since this is a score metric, so it needs to be maximised, rather than minimised.
    losses.append(loss)
    return losses, w