# Rational neural networks

We consider neural networks with rational activation functions. The choice of the nonlinear activation function in deep learning architectures is crucial and heavily impacts the performance of a neural network. We establish optimal bounds in terms of network complexity and prove that rational neural networks approximate smooth functions more efficiently than ReLU networks with exponentially smaller depth. The flexibility and smoothness of rational activation functions make them an attractive alternative to ReLU, as we demonstrate with numerical experiments.

For more information, please refer to

- Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend. "[Rational neural networks](https://arxiv.org/abs/2004.01902)." arXiv preprint arXiv:2004.01902 (2020).

## Dependencies

The Python packages used in the experiments are listed in the file `requirements.txt`. The main dependencies are TensorFlow (version 1.14) and Keras (version 2.2.4). Please note that we used the GPU version of TensorFlow.

## Content

- The Python file `RationalLayer.py` contains a TensorFlow implementation of a rational activation function, initialized by default to a type (3,2) approximation to the ReLU function. Other initial coefficients of the rational functions can be provided as parameters.
- Run the file `GAN/mnist-gan.py` to reproduce the GAN experiment.
- The files `KdV_relu.py`, `KdV_sine.py`, `KdV_rat.py` in the `Approximation_2d` folder train a ReLU, Sinusoid, and Rational network to interpolate a solution to the KdV equation.
- The MATLAB file `initial_rational_coeffs.m` can be used to compute different initialization coefficients of the rational activation functions (different type or approximation to other functions than ReLU). It requires the [Chebfun package](https://www.chebfun.org/).

## Citation

```
@article{boulle2020rational,
  title={Rational neural networks},
  author={Boull{\'e}, Nicolas and Nakatsukasa, Yuji and Townsend, Alex},
  journal={arXiv preprint arXiv:2004.01902},
  year={2020}
}
```