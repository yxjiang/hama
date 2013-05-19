package org.apache.hama.ml.perception;

import org.apache.hadoop.fs.Path;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;



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
public class SmallMultiLayerPerceptron extends MultiLayerPerceptron {

	/*	The path of the existing model	*/
	private Path modelPath;
	/*	The in-memory weight matrix	*/
	private DenseDoubleMatrix weightMat;

	
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

}
