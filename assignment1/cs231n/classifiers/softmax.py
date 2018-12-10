import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - max(scores)
        loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
        loss += loss_i
        for j in range(num_classes):
            softmax_output = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]
            else:
                dW[:, j] += softmax_output * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    score = X.dot(W)
    y_value = score[np.arange(num_train), y].reshape(num_train, 1)
    exp_array = np.exp(score)
    sum_array = np.sum(exp_array, axis=1).reshape(num_train, 1)
    loss += np.sum(np.log(sum_array) - y_value)
    # print(loss)
    # print(y_value)
    # loss = np.sum(np.sum(loss))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    margin = np.exp(score)
    margin[np.arange(num_train), y] += -1
    dw = X.T.dot(margin)
    dw /= num_train
    dw += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
