package org.apache.hama.ml.perception;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hama.ml.math.DoubleVector;

/**
 *	PerceptronBase defines the common behavior of all the concrete perceptrons. 
 *	
 */
public abstract class MultiLayerPerceptron {
	
	/*	The trainer for the model	*/
	protected PerceptronTrainer trainer;	
	/*	The file path that contains the model meta-data	*/
	protected Path modelPath;
	
	/*	Model meta-data	*/
	protected String MLPType;
  protected double learningRate;
	protected boolean regularization;
	protected double momentum;
	protected int numberOfLayers;
	protected String squashingFunctionName;
	protected String costFunctionName;
	protected int[] layerSizeArray;
	
	protected CostFunction costFunction;
	protected SquashingFunction squashingFunction;
	
	/**
	 * Initialize the MLP.
	 * @param modelPath							The location in file system to store the model.
	 * @param learningRate					Larger learningRate makes MLP learn more aggressive.
	 * @param regularization				Turn on regularization make MLP less likely to overfit.
	 * @param momentum							The momentum makes the historical adjust have affect to current adjust.
	 * @param squashingFunctionName	The name of squashing function.
	 * @param costFunctionName			The name of the cost function.
	 * @param layerSizeArray				The number of neurons for each layer. Note that the actual size of each layer is one more than the input size.
	 */
	public MultiLayerPerceptron(Path modelPath, double learningRate, boolean regularization, double momentum,
			String squashingFunctionName, String costFunctionName, int[] layerSizeArray) {
		this.modelPath = modelPath;
		this.learningRate = learningRate;
		this.regularization = regularization;	//	no regularization
		this.momentum = momentum;	//	no momentum
		this.squashingFunctionName = squashingFunctionName;
		this.costFunctionName = costFunctionName;
		this.layerSizeArray = layerSizeArray;
		this.numberOfLayers = this.layerSizeArray.length;
		
		//	hard-coded
		this.costFunction = new CostFunction();
		this.squashingFunction = new Sigmoid();
	}
	
	/**
	 * Initialize a multi-layer perceptron with existing model.
	 * @param modelPath		Location of existing model meta-data.
	 */
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
	public abstract DoubleVector output(DoubleVector featureVector) throws Exception;
	
	/**
	 * Read the model meta-data from the specified location.
	 * @throws IOException
	 */
	protected abstract void readFromModel() throws IOException;
	
	/**
	 * Write the model data to specified location.
	 * @throws IOException
	 */
	public abstract void writeModelToFile() throws IOException;
	
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

	public Path getModelPath() {
		return modelPath;
	}

	public String getMLPType() {
		return MLPType;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public boolean isRegularization() {
		return regularization;
	}

	public double getMomentum() {
		return momentum;
	}

	public int getNumberOfLayers() {
		return numberOfLayers;
	}

	public String getSquashingFunctionName() {
		return squashingFunctionName;
	}

	public String getCostFunctionName() {
		return costFunctionName;
	}

	public int[] getLayerSizeArray() {
		return layerSizeArray;
	}

}
