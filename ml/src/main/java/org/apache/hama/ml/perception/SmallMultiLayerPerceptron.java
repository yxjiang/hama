package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.BitSet;
import java.util.Map;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSPJob;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
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
public final class SmallMultiLayerPerceptron extends MultiLayerPerceptron implements Writable {
	
	/*	The in-memory weight matrix	*/
	private DoubleMatrix[] weightMatrix;
	
	/**
	 * {@inheritDoc}
	 */
	public SmallMultiLayerPerceptron(double learningRate, boolean regularization, 
			double momentum, String squashingFunctionName, String costFunctionName, int[] layerSizeArray) {
		super(learningRate, regularization, momentum, 
				squashingFunctionName, costFunctionName, layerSizeArray);
		this.MLPType = "SmallMLP";
		initializeWeightMatrix();
	}
	
	/**
	 * {@inheritDoc}
	 */
	public SmallMultiLayerPerceptron(String modelPath) {
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
		try {
			URI uri = new URI(modelPath);
			FileSystem fs = FileSystem.get(uri, conf);
			FSDataInputStream is = new FSDataInputStream(fs.open(new Path(modelPath)));
			this.readFields(is);
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Write the model to file.
	 * @throws IOException
	 */
	@Override
	public void writeModelToFile(String modelPath) throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		FSDataOutputStream stream = fs.create(new Path(modelPath), true);
		this.write(stream);
		stream.close();
	}
	
	@Override
	/**
	 * {@inheritDoc}
	 */
	public void train(Path dataInputPath, Map<String, String> trainingParams) 
			throws IOException, InterruptedException, ClassNotFoundException {
		// TODO Auto-generated method stub
		//	call a BSP job to train the model and then store the result into weightMat
		
		//	create the BSP training job
		Configuration conf = new Configuration();
		for (Map.Entry<String, String> entry : trainingParams.entrySet()) {
			conf.set(entry.getKey(), entry.getValue());
		}
		
		//	put model related parameters
		conf.set("modelPath", modelPath == null? "" : modelPath);
		if (modelPath == null || modelPath.trim().length() == 0) {	//	build model from scratch
			conf.set("learningRate", "" + this.learningRate);
			conf.set("regularization", "" + this.regularization);
			conf.set("momentum", "" + this.momentum);
			conf.set("squashingFunctionName", this.squashingFunctionName);
			conf.set("costFunctionName", this.costFunctionName);
			StringBuilder layerSizeArraySb = new StringBuilder();
			for (int layerSize : this.layerSizeArray) {
				layerSizeArraySb.append(layerSize);
				layerSizeArraySb.append(' ');
			}
			conf.set("layerSizeArray", layerSizeArraySb.toString());
		}
		
		HamaConfiguration hamaConf = new HamaConfiguration(conf);
		BSPJob job = new BSPJob(hamaConf, SmallMLPTrainer.class);
		job.setJobName("Small scale MLP training");
		job.setJarByClass(SmallMLPTrainer.class);
		job.setBspClass(SmallMLPTrainer.class);
		job.setInputPath(dataInputPath);
		job.setInputFormat(org.apache.hama.bsp.SequenceFileInputFormat.class);
		job.setInputKeyClass(LongWritable.class);
		job.setInputValueClass(VectorWritable.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(NullWritable.class);
		job.setOutputFormat(org.apache.hama.bsp.NullOutputFormat.class);
		
		int numTasks = conf.getInt("tasks", 1);
		job.setNumBspTask(numTasks);
		job.waitForCompletion(true);
	}
	
	
	/**
	 * The perceptron trainer for small scale MLP.
	 *
	 */
	private static class SmallMLPTrainer extends PerceptronTrainer {
		
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
	
}