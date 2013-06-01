package org.apache.hama.ml.perception;

/**
 * The cost function factory that generates the cost function
 * by name.
 */
public class CostFunctionFactory {

	/**
	 * Get the cost function according to the name.
	 * If no matched cost function is found, return the 
	 * SquaredError by default.
	 * @param name	The name of the cost function.
	 * @return	The cost function instance.
	 */
	public static CostFunction getCostFunction(String name) {
		if (name.equalsIgnoreCase("SquaredError")) {
			return new SquaredError();
		}
		else if (name.equalsIgnoreCase("LogisticError")) {
			return new LogisticCostFunction();
		}
		return new SquaredError();
	}
}
