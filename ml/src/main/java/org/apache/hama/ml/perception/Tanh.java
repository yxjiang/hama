package org.apache.hama.ml.perception;

/**
 * The hyperbolic tangent function.
 * It is used as a squashing function in multilayer perceptron. 
 *
 */
public class Tanh extends SquashingFunction {

	@Override
	public double calculate(double value) {
		return Math.tanh(value);
	}

	@Override
	public double getDerivative(double value) {
		return 1 - value * value;
	}

}
