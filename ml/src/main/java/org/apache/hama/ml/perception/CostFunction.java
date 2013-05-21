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
		return diff * diff;
	}
}
