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
	 * The calss label for the training data is represented in form of 
	 * {@link org.apache.hama.ml.math.DoubleVector}. 
	 * The interpretation of the vector is depend on the user. 
	 * Taking a 2-class classification task for example, the user can use one output
	 * neuron to represent the class label, where the value of the neural 0 and 1 
	 * indicate different class label. Also, the user can use two output neurons to 
	 * represent the output, where the vector [1, 0] and [0, 1] are used to indicate 
	 * these two classes.
	 * 
	 * @param trainingInstance	The training instance in vector format.
	 * @param	classLabel		The class label in form of {@link org.apache.hama.ml.math.DoubleVector}.
	 * The dimension of vector should be the same as the number of neurons in the output layer.
	 */
	public abstract void train(DoubleVector trainingInstance, DoubleVector classLabel);
	
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
	 * @param featureVector	The feature of an instance to feed the perceptron.
	 * @return	The results.
	 */
	public abstract DoubleVector output(DoubleVector featureVector);
}
