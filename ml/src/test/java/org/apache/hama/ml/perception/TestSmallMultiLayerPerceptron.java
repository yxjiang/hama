package org.apache.hama.ml.perception;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.writable.MatrixWritable;
import org.junit.Test;


public class TestSmallMultiLayerPerceptron {

	/**
	 * Write and read the parameters of MLP.
	 */
	@Test
	public void testWriteReadMLP() {
		Path modelPath = new Path("sampleModel.data");
		double learningRate = 0.5;
		boolean regularization = false;	//	no regularization
		double momentum = 0;	//	no momentum
		String squashingFunctionName = "Sigmoid";
		String costFunctionName = "SquareError";
		int[] layerSizeArray = new int[]{3, 2, 2, 3};
		MultiLayerPerceptron mlp = new SmallMultiLayerPerceptron(modelPath, learningRate, regularization, 
				momentum, squashingFunctionName, costFunctionName, layerSizeArray);
		try {
			mlp.writeModelToFile();
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
			fs.delete(modelPath, true);
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
		Path modelPath = new Path("sampleModel.data");
		Configuration conf = new Configuration();
		try {
			FileSystem fs = FileSystem.get(conf);
			FSDataOutputStream output = fs.create(modelPath);
			
			String MLPType = "SmallMLP";
			double learningRate = 0.5;
			boolean regularization = false;
			double momentum = 0;
			String squashingFunctionName = "Sigmoid";
			String costFunctionName = "SquareError";
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
			fs.delete(modelPath, true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
