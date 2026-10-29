from wk5_logistic import *

def calculate_loss_weighted(y, tx, w, bias):
    """compute the cost by negative log likelihood. Bias is the favor given towards the positive class. 0.50 is neutral

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    t1_f = (1-bias)*2
    t2_f = bias*2
    N = len(y)
    term1 = y.T @ np.log(sigmoid(tx @ w))
    term2 = (1 - y.T) @ np.log(1 - sigmoid(tx @ w))
    return float(((-1/N)*(t1_f*term1 + t2_f*term2))[0, 0])

def penalized_logistic_regression_weighted(y, tx, w, lambda_, bias):
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
    t1_f = (1-bias)
    t2_f = bias

    tx1 = tx[(y == 1)[0]]
    tx2 = tx[(y == 0)[0]]
    y1 = y[y == 1]
    y2 = y[y == 0]

    term1 = t1_f * tx1.T@(sigmoid(tx1 @ w) - y1)
    term2 = t2_f * tx2.T@(sigmoid(tx2 @ w) - y2)

    # This needs to be done, since len(term) doesn't work apparently
    if term1.shape[1] == 0:
        term1 = np.zeros(term2.shape)
    elif term2.shape[1] == 0:
        term2 = np.zeros(term1.shape)

    grad = (1/N)*(term1 + term2) + 2 * lambda_ * w
    grad = grad.reshape(len(grad), 1)
    loss = calculate_loss_weighted(y, tx, w, bias)
    return grad, loss