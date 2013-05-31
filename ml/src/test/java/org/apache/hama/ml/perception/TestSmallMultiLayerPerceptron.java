package org.apache.hama.ml.perception;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.writable.MatrixWritable;
import org.apache.hama.ml.writable.VectorWritable;
import org.junit.Ignore;
import org.junit.Test;


public class TestSmallMultiLayerPerceptron {

	/**
	 * Write and read the parameters of MLP.
	 */
	@Test
	public void testWriteReadMLP() {
		String modelPath = "sampleModel.data";
		double learningRate = 0.5;
		boolean regularization = false;	//	no regularization
		double momentum = 0;	//	no momentum
		String squashingFunctionName = "Sigmoid";
		String costFunctionName = "SquaredError";
		int[] layerSizeArray = new int[]{3, 2, 2, 3};
		MultiLayerPerceptron mlp = new SmallMultiLayerPerceptron(learningRate, regularization, 
				momentum, squashingFunctionName, costFunctionName, layerSizeArray);
		try {
			mlp.writeModelToFile(modelPath);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			//		read the meta-data
			Configuration conf = new Configuration();
			FileSystem fs = FileSystem.get(conf);
			mlp = new SmallMultiLayerPerceptron(modelPath);
			assertEquals("SmallMLP", mlp.getMLPType());
			assertEquals(learningRate, mlp.getLearningRate(), 0.001);
			assertEquals(regularization, mlp.isRegularization());
			assertEquals(layerSizeArray.length, mlp.getNumberOfLayers());
			assertEquals(momentum, mlp.getMomentum(), 0.001);
			assertEquals(squashingFunctionName, mlp.getSquashingFunctionName());
			assertEquals(costFunctionName, mlp.getCostFunctionName());
			assertArrayEquals(layerSizeArray, mlp.getLayerSizeArray());
			//		delete test file
			fs.delete(new Path(modelPath), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Test the output of an example MLP.
	 */
	@Test
	public void testOutput() {
		//	write the MLP meta-data manually
		String modelPath = "sampleModel.data";
		Configuration conf = new Configuration();
		try {
			FileSystem fs = FileSystem.get(conf);
			FSDataOutputStream output = fs.create(new Path(modelPath));
			
			String MLPType = "SmallMLP";
			double learningRate = 0.5;
			boolean regularization = false;
			double momentum = 0;
			String squashingFunctionName = "Sigmoid";
			String costFunctionName = "SquaredError";
			int[] layerSizeArray = new int[] {3, 2, 3, 3};
			int numberOfLayers = layerSizeArray.length;
			
			WritableUtils.writeString(output, MLPType);
			output.writeDouble(learningRate);
			output.writeBoolean(regularization);
			output.writeDouble(momentum);
			output.writeInt(numberOfLayers);
			WritableUtils.writeString(output, squashingFunctionName);
			WritableUtils.writeString(output, costFunctionName);
			
			//	write the number of neurons for each layer
			for (int i = 0; i < numberOfLayers; ++i) {
				output.writeInt(layerSizeArray[i]);
			}
			
			double[][] matrix01 = {	//	4 by 2
					{0.5, 0.2},
					{0.1, 0.1},
					{0.2, 0.5},
					{0.1, 0.5}
					};
			
			double[][] matrix12 = {	//	3 by 3
					{0.1, 0.2, 0.5},
					{0.2, 0.5, 0.2},
					{0.5, 0.5, 0.1}
			};
			
			double[][] matrix23 = {	//	4 by 3
					{0.2, 0.5, 0.2},
					{0.5, 0.1, 0.5},
					{0.1, 0.2, 0.1},
					{0.1, 0.2, 0.5}
			};
			
			DoubleMatrix[] matrices = {new DenseDoubleMatrix(matrix01), new DenseDoubleMatrix(matrix12), new DenseDoubleMatrix(matrix23)};
			for (DoubleMatrix mat : matrices) {
				MatrixWritable.write(mat, output);
			}
			output.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//	initial the mlp with existing model meta-data and get the output
		MultiLayerPerceptron mlp = new SmallMultiLayerPerceptron(modelPath);
		int[] sizes = mlp.getLayerSizeArray();
		DoubleVector input = new DenseDoubleVector(new double[]{1, 2, 3});
		try {
			DoubleVector result = mlp.output(input);
			assertArrayEquals(new double[]{0.6636557, 0.7009963, 0.7213835}, result.toArray(), 0.0001);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		//	delete meta-data
		try {
			FileSystem fs = FileSystem.get(conf);
			fs.delete(new Path(modelPath), true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Test the trainByInstance method.
	 */
	@Test
	public void testSingleInstanceTraining() {
		//	generate training data
		DoubleVector[] trainingData = new DenseDoubleVector[] {
				new DenseDoubleVector(new double[] {0, 0, 0}),
				new DenseDoubleVector(new double[] {0, 1, 1}),
				new DenseDoubleVector(new double[] {1, 0, 1}),
				new DenseDoubleVector(new double[] {1, 1, 0})
		};
		
		//	set parameters
		double learningRate = 0.6;
		boolean regularization = false;	//	no regularization
		double momentum = 0;	//	no momentum
		String squashingFunctionName = "Sigmoid";
		String costFunctionName = "SquaredError";
		int[] layerSizeArray = new int[]{2, 5, 1};
		SmallMultiLayerPerceptron mlp = new SmallMultiLayerPerceptron(learningRate, regularization, 
				momentum, squashingFunctionName, costFunctionName, layerSizeArray);
		
		try {
			//	train by multiple instances
			Random rnd = new Random();
			for (int i = 0; i < 30000; ++i) {
				DenseDoubleMatrix[] weightUpdates = mlp.trainByInstance(trainingData[rnd.nextInt(4)]);
				mlp.updateWeightMatrices(weightUpdates);
			}
			
//			System.out.printf("Weight matrices: %s\n", mlp.weightsToString(mlp.getWeightMatrices()));
			for (int i = 0; i < trainingData.length; ++i) {
				DenseDoubleVector testVec = (DenseDoubleVector)trainingData[i].slice(2);
				assertEquals(trainingData[i].toArray()[2], mlp.output(testVec).toArray()[0], 0.1);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Test the XOR problem.
	 */
	@Test
	public void testTrainingByXOR() {
		//	write in some training instances
		Configuration conf = new Configuration();
		String strDataPath = "hdfs://localhost:9000/tmp/xor";
		Path dataPath = new Path(strDataPath);
		
		//	generate training data
		DoubleVector[] trainingData = new DenseDoubleVector[] {
				new DenseDoubleVector(new double[] {0, 0, 0}),
				new DenseDoubleVector(new double[] {0, 1, 1}),
				new DenseDoubleVector(new double[] {1, 0, 1}),
				new DenseDoubleVector(new double[] {1, 1, 0})
		};
		
		try {
			URI uri = new URI(strDataPath);
			FileSystem hdfs = FileSystem.get(uri, conf);
			hdfs.delete(dataPath, true);
			if (!hdfs.exists(dataPath)) {
				hdfs.createNewFile(dataPath);
				SequenceFile.Writer writer = new SequenceFile.Writer(hdfs, conf, dataPath, 
																					LongWritable.class, VectorWritable.class);
				
				Random rnd = new Random();
				for (int i = 0; i < 1000; ++i) {
					VectorWritable vecWritable = new VectorWritable(trainingData[i % 4]);
					writer.append(new LongWritable(i), vecWritable);
				}
				writer.close();
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//	begin training
		String modelPath = "xorModel.data";
		double learningRate = 0.6;
		boolean regularization = false;	//	no regularization
		double momentum = 0;	//	no momentum
		String squashingFunctionName = "Tanh";
		String costFunctionName = "SquareError";
		int[] layerSizeArray = new int[]{2, 5, 1};
		SmallMultiLayerPerceptron mlp = new SmallMultiLayerPerceptron(learningRate, regularization, 
				momentum, squashingFunctionName, costFunctionName, layerSizeArray);
		
		Map<String, String> trainingParams = new HashMap<String, String>();
		trainingParams.put("training.iteration", "10000");
		trainingParams.put("training.mode", "minibatch.gradient.descent");
		trainingParams.put("training.batch.size", "100");
		trainingParams.put("tasks", "3");
		trainingParams.put("modelPath", modelPath);
		
		System.out.println("Before training");
		DenseDoubleMatrix[] matrices = mlp.getWeightMatrices();
		for (DenseDoubleMatrix m : matrices) {
			System.out.println();
			System.out.printf("%s\n", m.toString());
		}
		
		
		try {
			mlp.train(dataPath, trainingParams);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		System.out.println("After training");
		matrices = mlp.getWeightMatrices();
		for (DenseDoubleMatrix m : matrices) {
			System.out.printf("%s\n", m.toString());
		}
		
		//	test the model
		for (int i = 0; i < trainingData.length; ++i) {
			DenseDoubleVector testVec = (DenseDoubleVector)trainingData[i].slice(2);
			DenseDoubleVector expectedVec = (DenseDoubleVector)trainingData[i].slice(2, 3);
			try {
				DenseDoubleVector actual = (DenseDoubleVector)mlp.output(testVec);
				assertEquals(trainingData[i].toArray()[2], actual.get(0), 0.1);
				System.out.printf("Input: %s,\tExpected: %s,\tTest: %s\n", testVec, expectedVec, actual);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
}
