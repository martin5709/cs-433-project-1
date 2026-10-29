import numpy as np

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return (1 + np.exp(-t))**-1

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    N = len(y)
    term1 = y.T @ np.log(sigmoid(tx @ w))
    term2 = (1 - y.T) @ np.log(1 - sigmoid(tx @ w))
    return float(((-1/N)*(term1 + term2))[0, 0])

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************

    # Term 1 derivative: y_i * (1 - sigmoid(x_i^T w)) * x_i.T
    # Term 2 derivative: (1 - y_i) * (-sigmoid(x_i^T w)) * x_i.T
    N = len(y)
    return (1/N)*(tx.T@(sigmoid(tx @ w) - y))

def logistic_regression(y, tx, w):
    """return the gradient of the loss and the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        gradient: shape=(D, 1)
        loss: scalar number

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> loss, gradient, hessian = logistic_regression(y, tx, w)
    >>> round(loss, 8)
    0.62137268
    >>> gradient, hessian
    (array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]]), array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]]))
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    # ***************************************************
    return calculate_gradient(y, tx, w), calculate_loss(y, tx, w)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the gradient and the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    # ***************************************************

    # NOTE TO SELF: I don't think they want the hessian based on the docstring.
    N = len(y)
    grad = (1/N)*(tx.T@(sigmoid(tx @ w) - y)) + 2 * lambda_ * w
    loss = calculate_loss(y, tx, w)
    return grad, loss

def f1_logistic(y, tx, w, lambda_):
    """return the gradient and the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    # ***************************************************

    # NOTE TO SELF: I don't think they want the hessian based on the docstring.
    # Try to create some loss function that optimizes for F1 score (negative log F1).
    # NOTE: Not actually a part of week 5.
    N = len(y)
    py0 = np.sum(y == 0)/N # p(y = 0)
    py1 = np.sum(y == 1)/N # p(y = 1)

    def tsplit(arr):
        return arr[arr[:, -1] == 0], arr[arr[:, -1] == 1]

    txy0, txy1 = tsplit(np.concatenate((tx, y), axis=1))
    txy0 = txy0[:, 0:-1]
    txy1 = txy1[:, 0:-1]

    fwy0 = (1 + np.exp(-(txy0 @ w)))**-1
    gfwy0 = -1 * np.prod((1 + np.exp(-(txy0 @ w)))**-2) * (-txy0.T @ np.exp(-(txy0 @ w)))
    fwy1 = (1 + np.exp(-(txy1 @ w)))**-1
    gfwy1 = -1 * np.prod((1 + np.exp(-(txy1 @ w)))**-2) * (-txy1.T @ np.exp(-(txy1 @ w)))

    grad = gfwy1/np.prod(fwy1) - (py1*gfwy1 + py0*gfwy0)/(py1*(1 + np.prod(fwy1)) + py0*np.prod(fwy0)) - 2 * lambda_ * w # The last term is just to reduce the large weights. This needs to be in the opposite direction of the desired score.
    loss = np.log(2*np.prod(fwy1)*py1) - np.log(py1*(1+np.prod(fwy1)) + py0*np.prod(fwy0))

    if np.isnan(grad[0]):
        #print(txy0)
        #print(txy1)
        #print(fwy0)
        #print(fwy1)
        #assert False
        print("NaN detected.")
        return np.zeros(w.shape), 0

    return grad, loss
