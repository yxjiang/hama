package org.apache.hama.ml.perception;

/**
 * Square error cost function.
 * <pre>
 * cost(t, y) = 0.5 * (t - y)^2 
 * </pre>
 */
public class SquaredError extends CostFunction {

	@Override
	/**
	 * {@inheritDoc}
	 */
	public double getCost(double target, double actual) {
		double diff = target - actual;
		return 0.5 * diff * diff;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public double getPartialDerivative(double target, double actual) {
		return target - actual;
	}

}
