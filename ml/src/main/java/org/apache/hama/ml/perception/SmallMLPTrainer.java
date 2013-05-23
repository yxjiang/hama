package org.apache.hama.ml.perception;

import java.io.IOException;
import java.util.BitSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.writable.VectorWritable;

/**
 * The perceptron trainer for small scale MLP.
 *
 */
public class SmallMLPTrainer extends PerceptronTrainer {
	
	private static final Log LOG = LogFactory.getLog(SmallMLPTrainer.class);
	private BitSet statusSet;	//	used by master only, check whether all slaves finishes reading
	
	private int numRead = 0;
	private boolean terminated = false;
	
	private SmallMultiLayerPerceptron inMemoryPerceptron;
	
	@Override
	protected void extraSetup(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) {
		this.statusSet = new BitSet(peer.getConfiguration().getInt("tasks", 1));
		
		String modelPath = conf.get("modelPath");
		if (modelPath == null || modelPath.trim().length() == 0) {	//	build model from scratch
			String MLPType = conf.get("MLPType");
			double learningRate = Double.parseDouble(conf.get("learningRate"));
			boolean regularization = Boolean.parseBoolean(conf.get("regularization"));
			double momentum = Double.parseDouble(conf.get("momentum"));
			String squashingFunctionName = conf.get("squashingFunctionName");
			String costFunctionName = conf.get("costFunctionName");
			String[] layerSizeArrayStr = conf.get("layerSizeArray").trim().split(" ");
			int[] layerSizeArray = new int[layerSizeArrayStr.length];
			inMemoryPerceptron = new SmallMultiLayerPerceptron(learningRate, regularization, momentum, 
					squashingFunctionName, costFunctionName, layerSizeArray);
			LOG.info("Training model from scratch.");
		}
		else {	//	read model from existing data
			inMemoryPerceptron = new SmallMultiLayerPerceptron(modelPath);
			LOG.info("Training with existing model.");
		}
		
		
	}
	
	@Override
	protected void extraCleanup(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer
			) {
		LOG.info(String.format("Task %d read %d records.\n", peer.getPeerIndex(), this.numRead));
	}
	
	@Override
	public void bsp(BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer)
			throws IOException, SyncException, InterruptedException {
		// TODO Auto-generated method stub
		LOG.info("Start training...");
		if (trainingMode.equalsIgnoreCase("minibatch.gradient.descent")) {
			LOG.info("Training Mode: minibatch.gradient.descent");
			trainByMinibatch(peer);
		}
		
		LOG.info("Finished.");
	}
	
	/**
	 * Train the MLP with stochastic gradient descent.
	 * @param peer
	 * @throws IOException
	 * @throws SyncException
	 * @throws InterruptedException
	 */
	private void trainByMinibatch(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
					throws IOException, SyncException, InterruptedException {
		
		int maxIteration = conf.getInt("training.iteration", 1);
		LOG.info("Training Iteration: " + maxIteration);
		
		for (int i = 0; i < maxIteration; ++i) {
			peer.reopenInput();
			
			while (true) {
				//	master merges the updates
				if (peer.getPeerIndex() == 0) {
					mergeUpdate(peer);
				}
				peer.sync();
				
				//	update weights
				boolean terminate = updateWeights(peer);
				if (terminate) {
					break;
				}
				
				peer.sync();
			}
		}
		
	}
	
	/**
	 * Train the MLP with training data.
	 * @param peer
	 * @return Whether terminates.
	 * @throws IOException
	 */
	private boolean updateWeights(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
			throws IOException {
		//	receive update message
		if (peer.getNumCurrentMessages() > 0) {
			SmallMLPMessage message = (SmallMLPMessage)peer.getCurrentMessage();
			this.terminated = message.isTerminated();
			
			if (this.terminated) {
				return true;
			}
		}
		
		//	update weight according to training data
		int count = 0;
		LongWritable recordId = new LongWritable();
		VectorWritable trainingInstance = new VectorWritable();
		boolean hasMore = false;
		while (count++ < this.batchSize) {
			hasMore = peer.readNext(recordId, trainingInstance);
			++numRead;
			if (!hasMore) {
				break;
			}
		}
		
		LOG.info(String.format("Slave %d read %d records.\n", peer.getPeerIndex(), this.numRead));
		
		SmallMLPMessage message = new SmallMLPMessage(peer.getPeerIndex(), !hasMore, null);
		peer.send(peer.getPeerName(0), message);	//	send status to master
		
		return false;
	}
	
	/**
	 * Merge the updates from slaves task.
	 * @param peer
	 * @throws IOException 
	 */
	private void mergeUpdate(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
			throws IOException {
		
		while (peer.getNumCurrentMessages() > 0) {
			SmallMLPMessage message = (SmallMLPMessage)peer.getCurrentMessage();
			if (message.isTerminated()) {
				this.statusSet.set(message.getOwner());
			}
		}
		
		//	check if all tasks finishes reading data
		if (this.statusSet.cardinality() == conf.getInt("tasks", 1)) {
			this.terminated = true;
		}
		
		for (String peerName : peer.getAllPeerNames()) {
			SmallMLPMessage msg = new SmallMLPMessage(peer.getPeerIndex(), this.terminated, null);
			peer.send(peerName, msg);
		}
	}

}
