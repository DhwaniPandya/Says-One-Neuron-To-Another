from builtins import object
import numpy as np

from CNN_image_classification.layers import *
from CNN_image_classification.layer_utils import *


class ThreeLayerConvNet(object):
   

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype

       
        C, H, W = input_dim
        
        # Convolutional layer
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)

        # Hidden affine layer
        # Note that the width and height are preserved after the convolutional layer
        # 2x2 max pool makes the width and height reduce by half
        self.params['W2'] = np.random.randn(num_filters*(H//2)*(W//2), hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)

        # Output affine layer
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
         self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        
        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1,
            conv_param, pool_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_relu_pool_out,
            W2, b2)
        scores, scores_cache = affine_forward(affine_relu_out, W3, b3)

        
        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        daffine_relu_out, dW3, db3 = affine_backward(dscores, scores_cache)
        dconv_relu_pool_out, dW2, db2 = affine_relu_backward(daffine_relu_out, affine_relu_cache)
        dx, dW1, db1 = conv_relu_pool_backward(dconv_relu_pool_out, conv_relu_pool_cache)

        grads['W1'] = dW1 + self.reg * W1
        grads['W2'] = dW2 + self.reg * W2
        grads['W3'] = dW3 + self.reg * W3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        

        return loss, grads
