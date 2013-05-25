package org.apache.hama.ml.perception;

public abstract class CostFunction {
	
	/**
	 * Get the error evaluated by squared error.
	 * @param target	The target value.
	 * @param actual	The actual value.
	 * @return
	 */
	public abstract double getCost(double target, double actual);
	
	/**
	 * Get the partial derivative of squared error.
	 * @param target
	 * @param actual
	 * @return
	 */
	public abstract double getPartialDerivative(double target, double actual);

	
//	public static double squaredError(double target, double actual) {
//		double diff = target - actual;
//		return 0.5 * diff * diff;
//	}
//	
//	
//	public static double squaredErrorPartialDerivative(double target, double actual) {
//		return target- actual;
//	}
	
}
