package org.apache.hama.ml.perception;

import org.apache.hama.ml.math.DoubleVectorFunction;

/**
 * The squashing function to activate the neurons.
 * 
 */
public abstract class SquashingFunction implements DoubleVectorFunction {
	
	/**
   * Calculates the result with a given index and value of a vector.
   */
	public abstract double calculate(int index, double value);
	
	/**
	 * Apply the gradient descent to each of the elements in vector.
	 * @param vector
	 * @return
	 */
	public abstract double getDerivative(double value);
}
