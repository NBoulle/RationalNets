#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nicolas Boulle
"""

import tensorflow as tf

# Tensorflow >= 2.00
if int(tf.__version__[0]) >= 2:
    from tensorflow.keras.layers import Layer, InputSpec
    from tensorflow.keras import initializers, regularizers, constraints
# Tensorflow < 2.00
else:
    from keras.layers import Layer, InputSpec
    from keras import initializers, regularizers, constraints

class RationalLayer(Layer):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights of the numerator P.
        beta_initializer: initializer function for the weights of the denominator Q.
        alpha_regularizer: regularizer for the weights of the numerator P.
        beta_regularizer: regularizer for the weights of the denominator Q.
        alpha_constraint: constraint for the weights of the numerator P.
        beta_constraint: constraint for the weights of the denominator Q.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """

    def __init__(self, alpha_initializer=[1.1915, 1.5957, 0.5, 0.0218], beta_initializer=[2.383, 0.0, 1.0], 
                 alpha_regularizer=None, beta_regularizer=None, alpha_constraint=None, beta_constraint=None,
                 shared_axes=None, **kwargs):
        super(RationalLayer, self).__init__(**kwargs)
        self.supports_masking = True

        # Degree of rationals
        self.degreeP = len(alpha_initializer) - 1
        self.degreeQ = len(beta_initializer) - 1
        
        # Initializers for P
        self.alpha_initializer = [initializers.Constant(value=alpha_initializer[i]) for i in range(len(alpha_initializer))]
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        
        # Initializers for Q
        self.beta_initializer = [initializers.Constant(value=beta_initializer[i]) for i in range(len(beta_initializer))]
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        
        self.coeffsP = []
        for i in range(self.degreeP+1):
            # Add weight
            alpha_i = self.add_weight(shape=param_shape,
                                     name='alpha_%s'%i,
                                     initializer=self.alpha_initializer[i],
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
            self.coeffsP.append(alpha_i)
            
        # Create coefficients of Q
        self.coeffsQ = []
        for i in range(self.degreeQ+1):
            # Add weight
            beta_i = self.add_weight(shape=param_shape,
                                     name='beta_%s'%i,
                                     initializer=self.beta_initializer[i],
                                     regularizer=self.beta_regularizer,
                                     constraint=self.beta_constraint)
            self.coeffsQ.append(beta_i)
        
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
                    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
                    self.built = True

    def call(self, inputs, mask=None):
        # Evaluation of P
        outP = tf.math.polyval(self.coeffsP, inputs)
        # Evaluation of Q
        outQ = tf.math.polyval(self.coeffsQ, inputs)
        # Compute P/Q
        out = tf.math.divide(outP, outQ)
        return out

    def get_config(self):
        config = {
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(RationalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape