"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np 
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x , dtype=float)
    if x.ndim == 1:
        m = x.shape[0]
        n = 1
        for i in range ( m ) : 
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps 
            x_minus[i] -= eps 
            grad[i] = ( f(x_plus) - f( x_minus ) )  / ( 2 * eps )
        return grad 
    else:
        m, n = x.shape
        for i in range ( m ) : 
            for j in range ( n ) : 
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i][j] += eps 
                x_minus[i][j] -= eps 
                grad[i][j] = (f(x_plus) - f(x_minus)) / ( 2 * eps )
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    if analytic_grad.ndim == 1 : 
        e = abs ( analytic_grad[0] - numeric_grad[0] ) / max(abs(analytic_grad[0]) , abs(numeric_grad[0]) , tol )
        return e 
    m , n = analytic_grad.shape 
    e = np.zeros(m)
    for i in range (m) : 
        for j in range (n) : 
            a , nn = analytic_grad[i][j] , numeric_grad[i][j]
            e[i] = max(e[i] , abs(a-nn) / max ( abs ( a ) , abs ( nn) , tol) ) 
    res = np.max(e) 
    return res

# Step 3 - make_dense
import numpy as np

def make_dense(in_dim, out_dim, weight_init_fn):
    W, b = weight_init_fn(in_dim, out_dim)
    params = {"W": W, "b": b}

    def forward(x):
        y = x @ params["W"] + params["b"]
        return y, x

    def backward(dout, cache):
        x = cache

        if x.ndim == 1:
            dx = params["W"] @ dout
            dW = np.outer(x, dout)
            db = dout
        else:
            dx = dout @ params["W"].T
            dW = x.T @ dout
            db = np.sum(dout, axis=0)

        grads = {"W": dW, "b": db}
        return dx, grads

    return {
        "params": params,
        "forward": forward,
        "backward": backward
    }

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

