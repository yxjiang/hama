package org.apache.hama.ml.perception;

public class CostFunction {

	/**
	 * Get the error evaluated by squared error.
	 * @param target	The target value.
	 * @param actual	The actual value.
	 * @return
	 */
	public static double squaredError(double target, double actual) {
		double diff = target - actual;
		return 0.5 * diff * diff;
	}
	
	/**
	 * Get the partial derivative of squared error.
	 * @param target
	 * @param actual
	 * @return
	 */
	public static double squaredErrorPartialDerivative(double target, double actual) {
		return target- actual;
	}
	
}
