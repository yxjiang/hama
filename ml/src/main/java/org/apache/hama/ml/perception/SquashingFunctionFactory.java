package org.apache.hama.ml.perception;

/**
 * Get the squashing function according to the name.
 *
 */
public class SquashingFunctionFactory {
	
	/**
	 * Get the squashing function instance according to the name.
	 * If no matched squahsing function is found, return the sigmoid
	 * squashing function by default.
	 * @param name		The name of the squashing function.
	 * @return	The instance of the squashing function.
	 */
	public static SquashingFunction getSquashingFunction(String name) {
		if (name.equalsIgnoreCase("Sigmoid")) {
			return new Sigmoid();
		}
		else if (name.equalsIgnoreCase("Tanh")) {
			return new Tanh();
		}
		return new Sigmoid();
	}

}
