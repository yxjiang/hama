package org.apache.hama.ml.perception;

import org.apache.hama.ml.math.DoubleVector;

/**
 * The Sigmoid function 
 *
 */
public class Sigmoid extends SquashingFunction {

	@Override
	public double calculate(double value) {
		return 1.0 / (1 + Math.exp(-value));
	}

	@Override
	public double getDerivative(double value) {
//		double fx = this.calculate(0, value);
//		return fx * (1 - fx);
		return value * (1 - value);
	}
}
