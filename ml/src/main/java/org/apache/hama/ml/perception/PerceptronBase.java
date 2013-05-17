package org.apache.hama.ml.perception;

import org.apache.hama.ml.math.DoubleVector;

/**
 *	PerceptronBase defines the common behavior of all the concrete perceptrons. 
 *
 */
public interface PerceptronBase {
	
	/**
	 * Feed the training instance to the perceptron to update the weights.
	 * @param trainingInstance	The training instance in vector format.
	 * @param	classLabelIndex		The index of the class label.
	 */
	public abstract void train(DoubleVector trainingInstance, int classLabelIndex);
	
}
