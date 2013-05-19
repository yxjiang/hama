package org.apache.hama.ml.perception;

import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.io.LongWritable;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.math.DoubleVector;
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
public class SmallMultiLayerPerceptronBSP extends BSP<LongWritable, VectorWritable, LongWritable, VectorWritable, MLPMessage> implements PerceptronBase {
	
	/**
	 * Train this multi-layer perceptron with given portion of data.
	 */
	@Override
	public void bsp(BSPPeer<LongWritable, VectorWritable, LongWritable, VectorWritable, MLPMessage> peer)
			throws IOException, SyncException, InterruptedException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void train(DoubleVector trainingInstance, DoubleVector classLabel) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void loadModel(URI modelUri) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void saveModel(URI ModelUri) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public DoubleVector output(DoubleVector featureVector) {
		// TODO Auto-generated method stub
		return null;
	}

}
