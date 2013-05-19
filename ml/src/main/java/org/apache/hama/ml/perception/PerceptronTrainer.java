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
 *
 */
public abstract class PerceptronTrainer 
	extends BSP<LongWritable, VectorWritable, NeuronPairWritable, NullWritable, MLPMessage> {
	
	/*	The perceptron model	*/
	private MultiLayerPerceptron model;

	/**
   * {@inheritDoc}
   */
  @Override
  public abstract void bsp(BSPPeer<LongWritable, VectorWritable, NeuronPairWritable, NullWritable, MLPMessage> peer) 
  		throws IOException, SyncException, InterruptedException;
	
}
