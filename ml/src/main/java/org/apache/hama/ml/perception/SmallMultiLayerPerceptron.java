package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.writable.MatrixWritable;



/**
 * SmallMultiLayerPerceptronBSP is a kind of multilayer perceptron
 * whose parameters can be fit into the memory of a single machine.
 * This kind of model can be trained and used more efficiently than
 * the BigMultiLayerPerceptronBSP, whose parameters are distributedly
 * stored in multiple machines.
 *
 * In general, it it is a multilayer perceptron that consists
 * of one input layer, multiple hidden layer and one output layer.
 * 
 * The number of neurons in the input layer should be consistent with the
 * number of features in the training instance.
 * The number of neurons in the output layer
 *
 */
public final class SmallMultiLayerPerceptron extends MultiLayerPerceptron implements Writable {
	
	/*	The in-memory weight matrix	*/
	private DoubleMatrix[] weightMatrix;
	
	/**
	 * {@inheritDoc}
	 */
	public SmallMultiLayerPerceptron(Path modelPath, double learningRate, boolean regularization, 
			double momentum, String squashingFunctionName, String costFunctionName, int[] layerSizeArray) {
		super(modelPath, learningRate, regularization, momentum, 
				squashingFunctionName, costFunctionName, layerSizeArray);
		this.MLPType = "SmallMLP";
		initializeWeightMatrix();
	}
	
	/**
	 * {@inheritDoc}
	 */
	public SmallMultiLayerPerceptron(Path modelPath) {
		super(modelPath);
		if (modelPath != null) {
			try {
				this.readFromModel();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Initialize weight matrix using Gaussian distribution. 
	 */
	private void initializeWeightMatrix() {
		this.weightMatrix = new DenseDoubleMatrix[this.numberOfLayers - 1];
		//	each layer contains one bias neuron
		Random rnd = new Random();
		for (int i = 0; i < this.numberOfLayers - 1; ++i) {
			//	add weights for bias
			this.weightMatrix[i] = new DenseDoubleMatrix(this.layerSizeArray[i] + 1, this.layerSizeArray[i + 1]);
			int rowCount = this.weightMatrix[i].getRowCount();
			int colCount = this.weightMatrix[i].getColumnCount();
			for (int row = 0; row < rowCount; ++row) {
				for (int col = 0; col < colCount; ++col) {
					this.weightMatrix[i].set(row, col, rnd.nextGaussian());
				}
			}
		}
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public void train(Path dataInputPath) {
		// TODO Auto-generated method stub
		//	call a BSP job to train the model and then store the result into weightMat
		
		
	}

	@Override
	/**
	 * {@inheritDoc}
	 * The model meta-data is stored in memory.
	 */
	public DoubleVector output(DoubleVector featureVector) throws Exception {
		//	start from the first hidden layer
		double[] intermediateResults = new double[this.layerSizeArray[0] + 1];
		if (intermediateResults.length - 1 != featureVector.getDimension()) {
			throw new Exception("Input feature dimension incorrect! The dimension of input layer is " + 
					(this.layerSizeArray[0] - 1)  + ", but the dimension of input feature is " + featureVector.getDimension());
		}
		
		//	fill with input features
		intermediateResults[0] = 1.0;	//	bias
		for (int i = 0; i < featureVector.getDimension(); ++i) {
			intermediateResults[i + 1] = featureVector.get(i);
		}
		
		//	forward the intermediate results to next layer
		for (int fromLayer = 0; fromLayer < this.numberOfLayers - 1; ++fromLayer) {
			intermediateResults = forward(fromLayer, intermediateResults);
		}
		
		return new DenseDoubleVector(intermediateResults);
	}
	
	/**
	 * Calculate the intermediate results of layer fromLayer + 1.
	 * @param fromLayer		The index of layer that forwards the intermediate results from.
	 * @return
	 */
	private double[] forward(int fromLayer, double[] intermediateResult) {
		int toLayer = fromLayer + 1;
		double[] results = null;
		int offset = 0;
		
		if (toLayer < this.layerSizeArray.length - 1) {	//	add bias if it is not output layer
			results = new double[this.layerSizeArray[toLayer] + 1];
			offset = 1;
			results[0] = 1.0;	//	the bias
//			System.out.printf("From: %d to %d, # neurons at %d: %d\n", fromLayer, toLayer, toLayer, this.layerSizeArray[toLayer] + 1);
//			System.out.printf("Mat size: [%d, %d]\n", this.weightMatrix[fromLayer].getRowCount(), this.weightMatrix[fromLayer].getColumnCount());
		}
		else {
			results = new double[this.layerSizeArray[toLayer]];	//	no bias
//			System.out.println("Output layer.");
		}
		
		for (int neuronIdx = 0; neuronIdx < this.layerSizeArray[toLayer]; ++neuronIdx) {
			//	aggregate the results from previous layer
//			System.out.printf("For neuron %d\n", neuronIdx);
			for (int prevNeuronIdx = 0; prevNeuronIdx < this.layerSizeArray[fromLayer] + 1; ++prevNeuronIdx) {
//				System.out.printf("\t+ %f * %f", this.weightMatrix[fromLayer].get(prevNeuronIdx, neuronIdx), intermediateResult[prevNeuronIdx]);
				results[neuronIdx + offset] += this.weightMatrix[fromLayer].get(prevNeuronIdx, neuronIdx) * intermediateResult[prevNeuronIdx];
			}
//			System.out.printf("=%f\n", results[neuronIdx + offset]);
			results[neuronIdx + offset] = this.squashingFunction.calculate(0, results[neuronIdx + offset]);	//	calculate via squashing function
		}
//		System.out.printf("Result of layer: %d, %s\n", toLayer, Arrays.toString(results));
		
		return results;
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		this.MLPType = WritableUtils.readString(input);
		this.learningRate = input.readDouble();
		this.regularization = input.readBoolean();
		this.momentum = input.readDouble();
		this.numberOfLayers = input.readInt();
		this.squashingFunctionName = WritableUtils.readString(input);
		this.costFunctionName = WritableUtils.readString(input);
		//	read the number of neurons for each layer
		this.layerSizeArray = new int[this.numberOfLayers];
		for (int i = 0; i < numberOfLayers; ++i) {
			this.layerSizeArray[i] = input.readInt();
		}
		this.weightMatrix = new DenseDoubleMatrix[this.numberOfLayers - 1];
		for (int i = 0; i < numberOfLayers - 1; ++i)
			this.weightMatrix[i] = MatrixWritable.read(input);
		
		//	hard-coded
		this.squashingFunction = new Sigmoid();
		this.costFunction = new CostFunction();
	}

	@Override
	public void write(DataOutput output) throws IOException {
		WritableUtils.writeString(output, MLPType);
		output.writeDouble(learningRate);
		output.writeBoolean(regularization);
		output.writeDouble(momentum);
		output.writeInt(numberOfLayers);
		WritableUtils.writeString(output, squashingFunctionName);
		WritableUtils.writeString(output, costFunctionName);
		
		//	write the number of neurons for each layer
		for (int i = 0; i <this.numberOfLayers; ++i) {
			output.writeInt(this.layerSizeArray[i]);
		}
		for (int i = 0; i < numberOfLayers - 1; ++i) {
			MatrixWritable matrixWritable = new MatrixWritable(weightMatrix[i]);
			matrixWritable.write(output);
		}
	}

	/**
	 * Read the model meta-data from the specified location.
	 * @throws IOException
	 */
	@Override
	protected void readFromModel() throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		FSDataInputStream is = new FSDataInputStream(fs.open(modelPath));
		this.readFields(is);
	}
	
	/**
	 * Write the model to file.
	 * @throws IOException
	 */
	@Override
	public void writeModelToFile() throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		FSDataOutputStream stream = fs.create(modelPath, true);
		this.write(stream);
		stream.close();
	}
	
}
