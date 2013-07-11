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
package org.apache.hama.ml.regression;

import java.util.Random;

import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.assertArrayEquals;

/**
 * Test linear regression.
 * 
 */
public class TestLinearRegression {

  @Ignore
  @Test
  public void trainByOneInstance() {
    int featureDimension = 3;
    LinearRegression regression = new LinearRegression(featureDimension);
    regression.setLearningRate(0.1);
    double[] trainingInstance = { 2, 3, 5, 4.6 };

    DoubleMatrix[] delta = regression.trainByInstance(new DenseDoubleVector(
        trainingInstance));
    double[] expectedUpdates = new double[] { -0.45, -0.90, -1.35, -2.25 };
    DoubleVector vec = delta[0].getRowVector(0);
    // assertArrayEquals(expectedUpdates, vec.toArray(), 0.001);

    regression.updateWeightMatrices(delta);
    DoubleMatrix firstLayer = regression.getWeightsByLayer(0);
    DoubleVector firstLayerVec = firstLayer.getRowVector(0);
    double[] expectedNewWeights = new double[] { 0.05, -0.40, -0.85, -1.75 };
    // assertArrayEquals(expectedNewWeights, firstLayerVec.toArray(), 0.001);

    DoubleMatrix[] delta2 = regression.trainByInstance(new DenseDoubleVector(
        trainingInstance));
    DoubleVector vec2 = delta2[0].getRowVector(0);
    System.out.printf("Iteration 2: %s\n", vec2.toString());
    regression.updateWeightMatrices(delta2);

    DoubleMatrix[] delta3 = regression.trainByInstance(new DenseDoubleVector(
        trainingInstance));
    DoubleVector vec3 = delta3[0].getRowVector(0);
    System.out.printf("Iteration 3: %s\n", vec3.toString());
    regression.updateWeightMatrices(delta3);

    DoubleMatrix[] delta4 = regression.trainByInstance(new DenseDoubleVector(
        trainingInstance));
    DoubleVector vec4 = delta4[0].getRowVector(0);
    System.out.printf("Iteration 4: %s\n", vec4.toString());
    regression.updateWeightMatrices(delta4);

  }

  // @Ignore
  @Test
  public void testSimpleTraining() {
    int featureDimension = 3;
    LinearRegression regression = new LinearRegression(featureDimension);
    regression.setLearningRate(0.06);
    regression.setRegularizationWeight(0.1);
    Random rnd = new Random();
    // y = 0.3 * x1 + 0.5 * x2 + 0.5 * x3
    double[][] trainingData = { { 1, 1, 1, 1.3 + rnd.nextDouble() / 10 },
        { 2, 3, 5, 4.6 + rnd.nextDouble() / 10 },
        { 3, 5, 2, 4.4 + rnd.nextDouble() / 10 },
        { 5, 2, 1, 3 + rnd.nextDouble() / 10 },
        { 3, 1, 2, 2.4 + rnd.nextDouble() / 10 } };
    int iteration = 100;
    for (int i = 0; i < iteration; ++i) {
      DoubleMatrix[] updateMatrices = new DoubleMatrix[1];
      updateMatrices[0] = new DenseDoubleMatrix(1, featureDimension + 1);
      for (int j = 0; j < trainingData.length; ++j) {
        DoubleMatrix[] delta = regression
            .trainByInstance(new DenseDoubleVector(trainingData[j]));
        // aggregate
        updateMatrices[0] = updateMatrices[0].add(delta[0]);
      }
      updateMatrices[0] = updateMatrices[0].divide(trainingData.length);
      regression.updateWeightMatrices(updateMatrices);
      DoubleMatrix wMat = regression.getWeightsByLayer(0);
      for (int r = 0; r < wMat.getRowCount(); ++r) {
        for (int c = 0; c < wMat.getColumnCount(); ++c) {
          System.out.print(wMat.get(r, c) + " ");
        }
        System.out.println();
      }
    }
    DoubleVector instance = new DenseDoubleVector(new double[] { 10, 10, 10 });
    DoubleVector result = regression.getOutput(instance);
    System.out.printf("Final output:%f\n", result.get(0));
  }

}
