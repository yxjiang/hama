package org.apache.hama.ml.perception;

import java.net.URI;

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
	
	/**
	 * Directly load the existing model parameters from the specific location.
	 * @param modelUrl	The location to load the model parameters.
	 */
	public abstract void loadModel(URI modelUri);
	
	/**
	 * Store the trained model to specific location.
	 * The learnt parameter can be reloaded next time to avoid the training from scratch.
	 * @param modelUri	The location to save the model parameters.
	 */
	public abstract void saveModel(URI ModelUri);
	
	/**
	 * Get the output based on the input instance and the learned model.
	 * @param featureVector
	 * @return	The results.
	 */
	public abstract DoubleVector output(DoubleVector featureVector);
}
