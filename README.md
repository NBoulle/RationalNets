# Rational neural networks

We consider neural networks with rational activation functions. The choice of the nonlinear activation function in deep learning architectures is crucial and heavily impacts the performance of a neural network. We establish optimal bounds in terms of network complexity and prove that rational neural networks approximate smooth functions more efficiently than ReLU networks with exponentially smaller depth. The flexibility and smoothness of rational activation functions make them an attractive alternative to ReLU, as we demonstrate with numerical experiments.

For more information, please refer to

- Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend. "[Rational neural networks](https://arxiv.org/abs/2004.01902)." arXiv preprint arXiv:2004.01902 (2020).

## Dependencies

The Python packages used in the experiments are listed in the file `requirements.txt`. 
The main dependencies are TensorFlow (version 1.14) and Keras (version 2.2.4). 
We used the GPU version of TensorFlow: `tensorflow-gpu` for the GAN experiment.

## Content

- The Python file `RationalLayer.py` contains a TensorFlow implementation of a rational activation function, initialized by default to a type (3,2) approximation to the ReLU function. 
	- Import it using the line `from RationalLayer import RationalLayer` and add a rational layer with `model.add(RationalLayer())`
	
	- Other initial coefficients of the rational functions can be provided as parameters or the rational weigths can be shared across the nodes of the layer: `RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1,2,3])`

	- The MATLAB file `initial_rational_coeffs.m` computes different initialization coefficients for rational functions (rationals of different types or initialization near functions other than ReLU). It requires the [Chebfun package](https://www.chebfun.org/).

- In the `Approximation` folder, run `python3 main.py` to reproduce the approximation experiment of the paper.

- In the `GAN` folder, run `python3 mnist-gan.py` to reproduce the GAN experiment of the paper.

## Citation

```
@article{boulle2020rational,
  title={Rational neural networks},
  author={Boull{\'e}, Nicolas and Nakatsukasa, Yuji and Townsend, Alex},
  journal={arXiv preprint arXiv:2004.01902},
  year={2020}
}
```