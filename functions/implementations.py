# Code for Step 2

from wk2_gd import gradient_descent
from wk2_sgd import stochastic_gradient_descent
from wk3_copy import least_squares as lsq
from wk3_copy import ridge_regression as rr

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

# TODO
def logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass

# TODO
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass

