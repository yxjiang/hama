package org.apache.hama.ml.perception;

import org.apache.hama.ml.math.DoubleVector;

/**
 * The Sigmoid function 
 * <pre>
 * f(z) = 1 / (1 + e^{-z})
 * </pre>
 */
public class Sigmoid extends SquashingFunction {

	@Override
	public double calculate(int index, double value) {
		return 1.0 / (1 + Math.exp(-value));
	}

	@Override
	public double getDerivative(double value) {
		return value * (1 - value);
	}
}
