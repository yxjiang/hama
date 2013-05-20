package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.writable.MatrixWritable;
import org.apache.hama.ml.writable.VectorWritable;



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
public class SmallMultiLayerPerceptron extends MultiLayerPerceptron implements Writable {

	/*	The path of the existing model	*/
	private Path modelPath;
	
	/*	Meta-data	*/
	private static String MLPType = "SmallMLP";
	
	private double learningRate;
	private boolean regularization;
	private double momentum;
	private int numberOfLayers;
	private String squashingFunctionName;
	private String costFunctionName;
	private DoubleVector layerSizeVector;
	
	/*	The in-memory weight matrix	*/
	private DoubleMatrix weightMatrix;
	
	
	public SmallMultiLayerPerceptron(Path modelPath) {
		super(modelPath);
		// TODO Auto-generated constructor stub
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
	public DoubleVector output(DoubleVector featureVector) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		this.learningRate = input.readDouble();
		this.regularization = input.readBoolean();
		this.momentum = input.readDouble();
		this.numberOfLayers = input.readInt();
		this.squashingFunctionName = WritableUtils.readString(input);
		this.costFunctionName = WritableUtils.readString(input);
		this.layerSizeVector = VectorWritable.readVector(input);
		this.weightMatrix = MatrixWritable.read(input);
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeDouble(learningRate);
		output.writeBoolean(regularization);
		output.writeDouble(momentum);
		output.writeInt(numberOfLayers);
		WritableUtils.writeString(output, squashingFunctionName);
		WritableUtils.writeString(output, costFunctionName);
		VectorWritable.writeVector(layerSizeVector, output);
		MatrixWritable matrixWritable = new MatrixWritable(weightMatrix);
		matrixWritable.write(output);
	}

}
