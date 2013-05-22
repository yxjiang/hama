package org.apache.hama.ml.perception;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
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
	
	protected Configuration conf;
	protected int maxIteration;
	protected int batchSize;
	protected String trainingMode;
	
	@Override
	public void setup(BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
			throws IOException, SyncException, InterruptedException {
		conf = peer.getConfiguration();
		trainingMode = conf.get("training.mode");
		batchSize = conf.getInt("training.batch.size", 100);	//	mini-batch by default
		this.extraSetup(peer);
	}
	
	/**
	 * Handle extra setup for sub-classes.
	 * @param peer
	 * @throws IOException
	 * @throws SyncException
	 * @throws InterruptedException
	 */
	protected void extraSetup(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
					throws IOException, SyncException, InterruptedException {
	}
	
	/**
   * {@inheritDoc}
   */
  @Override
  public abstract void bsp(BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
  		throws IOException, SyncException, InterruptedException;
	
  @Override
  public void cleanup(BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
			throws IOException {
  	
  	this.extraCleanup(peer);
  }
  
  /**
	 * Handle extra cleanup for sub-classes.
	 * @param peer
	 * @throws IOException
	 * @throws SyncException
	 * @throws InterruptedException
	 */
	protected void extraCleanup(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
					throws IOException {
	}
  
}
