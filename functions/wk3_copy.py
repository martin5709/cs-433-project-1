import numpy as np

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    N = len(y)
    loss = (1/(2*N))*((y - tx @ w).T @ (y - tx @ w))

    return w, float(loss)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N = len(y)
    D = len(tx.T)
    w = np.linalg.solve(tx.T @ tx + lambda_*2*N*np.identity(D), tx.T @ y)
    loss = (1/(2*N))*((y - tx @ w).T @ (y - tx @ w)) + lambda_ * np.linalg.norm(w)**2
    return w, loss # In the exercises, it would fail if one also returns the loss, but here they require it according to the project document.