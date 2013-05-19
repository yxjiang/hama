package org.apache.hama.ml.perception;

import java.net.URI;

import org.apache.hadoop.fs.Path;
import org.apache.hama.ml.math.DoubleVector;

/**
 *	PerceptronBase defines the common behavior of all the concrete perceptrons. 
 *	
 */
public abstract class MultiLayerPerceptron {
	
	/*	The trainer for the model	*/
	private PerceptronTrainer trainer;	
	/*	The file path that contains the model meta-data	*/
	private Path modelPath;
	
	public MultiLayerPerceptron(Path modelPath) {
		this.modelPath = modelPath;
	}
	
	/**
	 * Train the model with given data.
	 * This method invokes a perceptron training BSP task to train the model.
	 * It then write the model to modelPath.
	 * @param dataInputPath 	The path of the data.
	 */
	public abstract void train(Path dataInputPath);
	
	/**
	 * Get the output based on the input instance and the learned model.
	 * @param featureVector	The feature of an instance to feed the perceptron.
	 * @return	The results.
	 */
	public abstract DoubleVector output(DoubleVector featureVector);
	
//	/**
//	 * Feed the training instance to the perceptron to update the weights.
//	 * The calss label for the training data is represented in form of 
//	 * {@link org.apache.hama.ml.math.DoubleVector}. 
//	 * The interpretation of the vector is depend on the user. 
//	 * Taking a 2-class classification task for example, the user can use one output
//	 * neuron to represent the class label, where the value of the neural 0 and 1 
//	 * indicate different class label. Also, the user can use two output neurons to 
//	 * represent the output, where the vector [1, 0] and [0, 1] are used to indicate 
//	 * these two classes.
//	 * 
//	 * @param trainingInstance	The training instance in vector format.
//	 * @param	classLabel		The class label in form of {@link org.apache.hama.ml.math.DoubleVector}.
//	 * The dimension of vector should be the same as the number of neurons in the output layer.
//	 */
//	protected abstract void train(DoubleVector trainingInstance, DoubleVector classLabel);

}
