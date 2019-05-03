from builtins import range
import numpy as np

def relu_forward(x):
    out = None
    
    out = np.maximum(x, 0)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    
    mask = x > 0
    dx = dout * mask

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        
        mu = np.mean(x, axis=0)

        xmu = x - mu
        sq = xmu ** 2
        var = np.var(x, axis=0)

        sqrtvar = np.sqrt(var + eps)
        ivar = 1./sqrtvar
        xhat = xmu * ivar

        out = gamma * xhat + beta

        cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
    elif mode == 'test':
        
        x_normalize = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalize + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    
    N, D = dout.shape

    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout*xhat, axis=0)
    dxhat = dout * gamma

    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar

    dsqrtvar = -1. / (sqrtvar**2) * divar
    dvar = 0.5 * 1. / np.sqrt(var+eps) * dsqrtvar

    dsq = 1. / N * np.ones((N, D)) * dvar
    dxmu2 = 2 * xmu * dsq

    dx1 = dxmu1 + dxmu2

    dmu = -1 * np.sum(dx1, axis=0)

    dx2 = 1. / N * np.ones((N, D)) * dmu

    dx = dx1 + dx2 

    
    return dx, dgamma, dbeta


def conv_forward_naive(x, w, b, conv_param):
        
    out = None
   
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride 
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    for n in range(N):
    	for f in range(F):
    		for h_out in range(H_out):
    			for w_out in range(W_out):
    				out[n, f, h_out, w_out] = np.sum(
    					x_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW]*w[f, :]) + b[f]

   
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    
    dx, dw, db = None, None, None
    
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
    	for f in range(F):
    		db[f] += np.sum(dout[n, f])
    		for h_out in range(H_out):
    			for w_out in range(W_out):
    				dw[f] += x_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] * \
    				dout[n, f, h_out, w_out]
    				dx_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] += w[f] * \
    				dout[n, f, h_out, w_out]

    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    
    out = None
    
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
    	for h_out in range(H_out):
    		for w_out in range(W_out):
    			out[n, :, h_out, w_out] = np.max(x[n, :, h_out*stride:h_out*stride+pool_height,
    				w_out*stride:w_out*stride+pool_width], axis=(1, 2)) 
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
   
    dx = None
    
    
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)

    for n in range(N):
    	for c in range(C):
    		for h in range(H_out):
    			for w in range(W_out):
    				
    				ind = np.unravel_index(np.argmax(x[n, c, h*stride:h*stride+pool_height,
    					w*stride:w*stride+pool_width], axis=None), (pool_height, pool_width))
    				
    				dx[n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width][ind] = \
    				dout[n, c, h, w]

   
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
   
    out, cache = None, None

    
    N, C, H, W = x.shape

    # Reshape x to N*H*W * C to call batch normalization
    x_new = np.reshape(np.transpose(x, (0, 2, 3, 1)), (-1, C))

    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    
    # Reshape out to (N, C, H, W)
    out = np.transpose(np.reshape(out, (N, H, W, C)), (0, 3, 1, 2))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    
    dx, dgamma, dbeta = None, None, None

    
    N, C, H, W = dout.shape

    # Reshape dout to N*H*W * C to call batch normalization
    dout_new = np.reshape(np.transpose(dout, (0, 2, 3, 1)), (-1, C))

    dx, dgamma, dbeta = batchnorm_backward_alt(dout_new, cache)

    # Reshape dx to (N, C, H, W)
    dx = np.transpose(np.reshape(dx, (N, H, W, C)), (0, 3, 1, 2))

 
    return dx, dgamma, dbeta



def svm_loss(x, y):
   #dx is gradient of loss with respect to x
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
   
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def batchnorm_backward_alt(dout, cache):
   
   
    dx, dgamma, dbeta = None, None, None
   
    
    N, D = dout.shape
    
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    dxhat = dout * gamma
    
    dx = 1.0/N * ivar * (N*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat*dout, axis=0)

    return dx, dgamma, dbeta

def affine_forward(x, w, b):
    #Computes the forward pass for an affine (fully-connected) layer.

    out = None
    
    out = x.reshape(x.shape[0], -1).dot(w) + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    
    # Computes the backward pass for an affine layer.

    x, w, b = cache
    dx, dw, db = None, None, None
    
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

