package org.apache.hama.ml.perception;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.Test;


public class TestSmallMultiLayerPerceptron {

	/**
	 * Write and read the parameters of MLP.
	 */
	@Test
	public void testWriteReadMLP() {
		Path modelPath = new Path("model.data");
		double learningRate = 0.5;
		boolean regularization = false;	//	no regularization
		int numberOfLayers = 3;
		double momentum = 0;	//	no momentum
		String squashingFunctionName = "Sigmoid";
		String costFunctionName = "MSE";
		int[] layerSizeArray = new int[]{3, 2, 2};
		MultiLayerPerceptron mlp = new SmallMultiLayerPerceptron(modelPath, learningRate, regularization, 
				numberOfLayers, momentum, squashingFunctionName, costFunctionName, layerSizeArray);
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
			assertEquals(learningRate, mlp.getLearningRate(), 0.001);
			assertEquals(regularization, mlp.isRegularization());
			assertEquals(numberOfLayers, mlp.getNumberOfLayers());
			assertEquals(momentum, mlp.getMomentum(), 0.001);
			assertEquals(squashingFunctionName, mlp.getSquashingFunctionName());
			assertEquals(costFunctionName, mlp.getCostFunctionName());
			assertArrayEquals(new int[]{3 + 1, 2 + 1, 2 + 1}, mlp.getLayerSizeArray());
			//		delete test file
			fs.delete(modelPath, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
