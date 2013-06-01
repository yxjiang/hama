package org.apache.hama.ml.perception;

/**
 * Teh common interface for cost functions.
 *
 */
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
	
}
