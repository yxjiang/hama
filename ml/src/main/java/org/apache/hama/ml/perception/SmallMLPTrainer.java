package org.apache.hama.ml.perception;

import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.writable.VectorWritable;

/**
 * The perceptron trainer for small scale MLP.
 *
 */
public class SmallMLPTrainer extends PerceptronTrainer {
	
	private static final Log LOG = LogFactory.getLog(SmallMLPTrainer.class);
	/*	used by master only, check whether all slaves finishes reading	*/
	private BitSet statusSet;	
	
	private int numTrainingInstanceRead = 0;
	/*	Once reader reaches the EOF, the training procedure would be terminated	*/
	private boolean terminateTraining = false;
	
	private SmallMultiLayerPerceptron inMemoryPerceptron;
	
	
	private int[] layerSizeArray;
	
	
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
			this.layerSizeArray = new int[layerSizeArrayStr.length];
			for (int i = 0; i < this.layerSizeArray.length; ++i) {
				this.layerSizeArray[i] = Integer.parseInt(layerSizeArrayStr[i]);
			}
			
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
		LOG.info(String.format("Task %d totally read %d records.\n", peer.getPeerIndex(), this.numTrainingInstanceRead));
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
			LOG.info(String.format("Iteration [%d] begins...", i));
			peer.reopenInput();
			//	reset
			if (peer.getPeerIndex() == 0) {
				this.statusSet = new BitSet(peer.getConfiguration().getInt("tasks", 1));
			}
			this.terminateTraining = false;
			peer.sync();
				
			while (true) {
				//	master merges the updates
				if (peer.getPeerIndex() == 0) {
					mergeUpdate(peer);
				}
				peer.sync();
				
				//	each slate task updates weights according to training data
				boolean terminate = updateWeights(peer);
				peer.sync();
				
				if (terminate) {
					break;
				}
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
		//	receive update message sent by master
		if (peer.getNumCurrentMessages() > 0) {
			SmallMLPMessage message = (SmallMLPMessage)peer.getCurrentMessage();
			this.terminateTraining = message.isTerminated();
			//	each slave renew its weight matrices
			this.inMemoryPerceptron.setWeightMatrices(message.getWeightsUpdatedMatrices());
			if (this.terminateTraining) {
				return true;
			}
		}
		
		//	update weight according to training data
		DenseDoubleMatrix[] weightUpdates = this.initWeightMatrices();
		
		int count = 0;
		LongWritable recordId = new LongWritable();
		VectorWritable trainingInstance = new VectorWritable();
		boolean hasMore = false;
		while (count++ < this.batchSize) {
			hasMore = peer.readNext(recordId, trainingInstance);
			
			try {
				DenseDoubleMatrix[] singleTrainingInstanceUpdates = this.inMemoryPerceptron.trainByInstance(trainingInstance.getVector());
				//	aggregate the updates
				for (int m = 0; m < weightUpdates.length; ++m) {
					weightUpdates[m] = (DenseDoubleMatrix)weightUpdates[m].add(singleTrainingInstanceUpdates[m]);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			++numTrainingInstanceRead;
			if (!hasMore) {
				break;
			}
		}
		
		//	calculate the local mean (the mean of the local batch) of weight updates
		for (int m = 0; m < weightUpdates.length; ++m) {
			weightUpdates[m] = (DenseDoubleMatrix)weightUpdates[m].divide(this.batchSize);
		}
		
		LOG.info(String.format("Task %d has read %d records.\n", peer.getPeerIndex(), this.numTrainingInstanceRead));
		
		//	send the weight updates to master task
		SmallMLPMessage message = new SmallMLPMessage(peer.getPeerIndex(), !hasMore, weightUpdates);
		peer.send(peer.getPeerName(0), message);	//	send status to master
		
		return !hasMore;
	}
	
	/**
	 * Merge the updates from slaves task.
	 * @param peer
	 * @throws IOException 
	 */
	private void mergeUpdate(
			BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer) 
			throws IOException {
		//	initialize the cache
		DenseDoubleMatrix[] weightUpdateCache = this.initWeightMatrices();
		
		int numOfPartitions = peer.getNumCurrentMessages();
		
		while (peer.getNumCurrentMessages() > 0) {
			SmallMLPMessage message = (SmallMLPMessage)peer.getCurrentMessage();
			if (message.isTerminated()) {
				this.statusSet.set(message.getOwner());
			}
			//	aggregates the weights
			DenseDoubleMatrix[] weightUpdates = message.getWeightsUpdatedMatrices();
			for (int m = 0; m < weightUpdateCache.length; ++m) {
				weightUpdateCache[m] = (DenseDoubleMatrix)weightUpdateCache[m].add(weightUpdates[m]);
			}
		}
		
		//	calculate the global mean (the mean of batches from all slave tasks) of the weight updates
		for (int m = 0; m < weightUpdateCache.length; ++m) {
			weightUpdateCache[m] = (DenseDoubleMatrix)weightUpdateCache[m].divide(numOfPartitions);
		}
		
		//	check if all tasks finishes reading data
		if (this.statusSet.cardinality() == conf.getInt("tasks", 1)) {
			this.terminateTraining = true;
		}
		
		LOG.info("Master: Weight update finishes.");
		
		//	update the weight matrices
		this.inMemoryPerceptron.updateWeightMatrices(weightUpdateCache);
		
		for (String peerName : peer.getAllPeerNames()) {
			SmallMLPMessage msg = new SmallMLPMessage(peer.getPeerIndex(), this.terminateTraining, 
					this.inMemoryPerceptron.getWeightMatrices());
			peer.send(peerName, msg);
		}
		LOG.info("Master: Broadcast updated weight matrix finishes.");
		
	}
	
	/**
	 * Initialize the weight matrices.
	 */
	private DenseDoubleMatrix[] initWeightMatrices() {
		DenseDoubleMatrix[] weightUpdateCache = new DenseDoubleMatrix[this.layerSizeArray.length - 1];
		//	initialize weight matrix each layer
		for (int i = 0; i < weightUpdateCache.length; ++i) {
			weightUpdateCache[i] = new DenseDoubleMatrix(this.layerSizeArray[i] + 1, this.layerSizeArray[i + 1]);
		}
		return weightUpdateCache;
	}
	
	/**
	 * Print out the weights.
	 * @param mat
	 * @return
	 */
	private static String weightsToString(DenseDoubleMatrix[] mat) {
		StringBuilder sb = new StringBuilder();
		
		for (int i = 0; i < mat.length; ++i) {
			sb.append(String.format("Matrix [%d]\n", i));
			double[][] values = mat[i].getValues();
			for (int d = 0; d < values.length; ++d) {
				sb.append(Arrays.toString(values[d]));
			}
			sb.append('\n');
		}
		return sb.toString();
	}

}
