package org.apache.hama.ml.perception;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.writable.VectorWritable;

/**
 * The trainer that is used to train the perceptron with BSP.
 * The trainer would read the training data and obtain the trained parameters of the model.
 */
public abstract class PerceptronTrainer 
	extends BSP<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> {
	
	/*	The perceptron model	*/
	private MultiLayerPerceptron model;
	
	public PerceptronTrainer(MultiLayerPerceptron model) {
		this.model = model;
	}
	
	/**
   * {@inheritDoc}
   */
  @Override
  public abstract void bsp(BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
  		throws IOException, SyncException, InterruptedException;
	
}
