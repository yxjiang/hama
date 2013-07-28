/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hama.ml.ann;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;

import org.apache.hama.ml.ann.AbstractLayeredNeuralNetwork.TrainingMethod;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.math.FunctionFactory;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Test the functionality of SmallLayeredNeuralNetwork.
 * 
 */
public class TestSmallLayeredNeuralNetwork {
  
  @Test
  public void testReadWrite() {
    SmallLayeredNeuralNetwork ann = new SmallLayeredNeuralNetwork();
    ann.addLayer(2, false,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.addLayer(5, false,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.addLayer(1, true,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction("SquaredError"));
    double learningRate = 0.2;
    ann.setLearningRate(learningRate);
    double momentumWeight = 0.5;
    ann.setMomentumWeight(momentumWeight);
    double regularizationWeight = 0.05;
    ann.setRegularizationWeight(regularizationWeight);
    // intentionally initialize all weights to 0.5
    DoubleMatrix[] matrices = new DenseDoubleMatrix[2];
    matrices[0] = new DenseDoubleMatrix(5, 3, 0.2);
    matrices[1] = new DenseDoubleMatrix(1, 6, 0.8);
    ann.setWeightMatrices(matrices);
    
    //  write to file
    String modelPath = "tmp/testReadWrite";
    ann.setModelPath(modelPath);
    try {
      ann.writeModelToFile();
    } catch (IOException e) {
      e.printStackTrace();
    }
    
    // read from file
    SmallLayeredNeuralNetwork annCopy = new SmallLayeredNeuralNetwork(modelPath);
    assertEquals(annCopy.getClass().getSimpleName(), annCopy.getModelType());
    assertEquals(modelPath, annCopy.getModelPath());
    assertEquals(learningRate, annCopy.getLearningRate(), 0.000001);
    assertEquals(momentumWeight, annCopy.getMomemtumWeight(), 0.000001);
    assertEquals(regularizationWeight, annCopy.getRegularizationWeight(), 0.000001);
    
    // compare weights
    DoubleMatrix[] weightsMatrices = annCopy.getWeightMatrices();
    for (int i = 0; i < weightsMatrices.length; ++i) {
      DoubleMatrix expectMat = matrices[i];
      DoubleMatrix actualMat = weightsMatrices[i];
      for (int j = 0; j < expectMat.getRowCount(); ++j) {
        for (int k = 0; k < expectMat.getColumnCount(); ++k) {
          assertEquals(expectMat.get(j, k), actualMat.get(j, k), 0.000001);
        }
      }
    }
  }

  @Test
  /**
   * Test the forward functionality.
   */
  public void testOutput() {
    // first network
    SmallLayeredNeuralNetwork ann = new SmallLayeredNeuralNetwork();
    ann.addLayer(2, false,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.addLayer(5, false,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.addLayer(1, true,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction("SquaredError"));
    ann.setLearningRate(0.1);
    // intentionally initialize all weights to 0.5
    DoubleMatrix[] matrices = new DenseDoubleMatrix[2];
    matrices[0] = new DenseDoubleMatrix(5, 3, 0.5);
    matrices[1] = new DenseDoubleMatrix(1, 6, 0.5);
    ann.setWeightMatrices(matrices);

    double[] arr = new double[] { 0, 1 };
    DoubleVector training = new DenseDoubleVector(arr);
    DoubleVector result = ann.getOutput(training);
    assertEquals(1, result.getDimension());
    assertEquals(3, result.get(0), 0.000001);

    // second network
    SmallLayeredNeuralNetwork ann2 = new SmallLayeredNeuralNetwork();
    ann2.addLayer(2, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann2.addLayer(3, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann2.addLayer(1, true, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann2.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction("SquaredError"));
    ann2.setLearningRate(0.3);
    // intentionally initialize all weights to 0.5
    DoubleMatrix[] matrices2 = new DenseDoubleMatrix[2];
    matrices2[0] = new DenseDoubleMatrix(3, 3, 0.5);
    matrices2[1] = new DenseDoubleMatrix(1, 4, 0.5);
    ann2.setWeightMatrices(matrices2);

    double[] test = { 0, 0 };
    double[] result2 = { 0.807476 };

    DoubleVector vec = ann2.getOutput(new DenseDoubleVector(test));
    assertArrayEquals(result2, vec.toArray(), 0.000001);

    SmallLayeredNeuralNetwork ann3 = new SmallLayeredNeuralNetwork();
    ann3.addLayer(2, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann3.addLayer(3, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann3.addLayer(1, true, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann3.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction("SquaredError"));
    ann3.setLearningRate(0.3);
    // intentionally initialize all weights to 0.5
    DoubleMatrix[] initMatrices = new DenseDoubleMatrix[2];
    initMatrices[0] = new DenseDoubleMatrix(3, 3, 0.5);
    initMatrices[1] = new DenseDoubleMatrix(1, 4, 0.5);
    ann3.setWeightMatrices(initMatrices);

    double[] instance = { 0, 1 };
    DoubleVector output = ann3.getOutput(new DenseDoubleVector(instance));
    assertEquals(0.8315410, output.get(0), 0.000001);
  }

  @Test
  public void testXORlocal() {
    SmallLayeredNeuralNetwork ann = new SmallLayeredNeuralNetwork();
    ann.addLayer(2, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann.addLayer(3, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann.addLayer(1, true, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction("SquaredError"));
    ann.setLearningRate(0.8);
    ann.setMomentumWeight(0.0);

    int iterations = 15000; // iteration should be set to a very large number
    double[][] instances = { { 0, 1, 1 }, { 0, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 } };
    for (int i = 0; i < iterations; ++i) {
      DoubleMatrix[] matrices = null;
      for (int j = 0; j < instances.length; ++j) {
        matrices = ann.trainByInstance(
            new DenseDoubleVector(instances[j % instances.length]), TrainingMethod.GRADIATE_DESCENT);
        ann.updateWeightMatrices(matrices);
      }
    }

    for (int i = 0; i < instances.length; ++i) {
      DoubleVector input = new DenseDoubleVector(instances[i]).slice(2);
      // the expected output is the last element in array
      double result = instances[i][2];
      assertEquals(result, ann.getOutput(input).get(0), 0.05);
    }

    // write model into file and read out
    String modelPath = "tmp/testXORLocal";
    ann.setModelPath(modelPath);
    try {
      ann.writeModelToFile();
    } catch (IOException e) {
      e.printStackTrace();
    }
    SmallLayeredNeuralNetwork annCopy = new SmallLayeredNeuralNetwork(modelPath);
    // test on instances
    for (int i = 0; i < instances.length; ++i) {
      DoubleVector input = new DenseDoubleVector(instances[i]).slice(2);
      // the expected output is the last element in array
      double result = instances[i][2];
      assertEquals(result, annCopy.getOutput(input).get(0), 0.05);
    }
  }
  
  @Test
  public void testXORwithMomentum() {
    SmallLayeredNeuralNetwork ann = new SmallLayeredNeuralNetwork();
    ann.addLayer(2, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann.addLayer(3, false, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann.addLayer(1, true, FunctionFactory.createDoubleFunction("Sigmoid"));
    ann.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction("SquaredError"));
    ann.setLearningRate(0.8);
    ann.setMomentumWeight(0.5);

    int iterations = 500; // iteration should be set to a very large number
    double[][] instances = { { 0, 1, 1 }, { 0, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 } };
    for (int i = 0; i < iterations; ++i) {
      DoubleMatrix[] matrices = null;
      for (int j = 0; j < instances.length; ++j) {
        matrices = ann.trainByInstance(
            new DenseDoubleVector(instances[j % instances.length]), TrainingMethod.GRADIATE_DESCENT);
        ann.updateWeightMatrices(matrices);
      }
    }

    for (int i = 0; i < instances.length; ++i) {
      DoubleVector input = new DenseDoubleVector(instances[i]).slice(2);
      // the expected output is the last element in array
      double result = instances[i][2];
      assertEquals(result, ann.getOutput(input).get(0), 0.05);
    }

    // write model into file and read out
    String modelPath = "tmp/testXORLocal";
    ann.setModelPath(modelPath);
    try {
      ann.writeModelToFile();
    } catch (IOException e) {
      e.printStackTrace();
    }
    SmallLayeredNeuralNetwork annCopy = new SmallLayeredNeuralNetwork(modelPath);
    // test on instances
    for (int i = 0; i < instances.length; ++i) {
      DoubleVector input = new DenseDoubleVector(instances[i]).slice(2);
      // the expected output is the last element in array
      double result = instances[i][2];
      assertEquals(result, annCopy.getOutput(input).get(0), 0.05);
    }
  }

}
