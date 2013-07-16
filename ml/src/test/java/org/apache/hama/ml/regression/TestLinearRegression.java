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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.writable.VectorWritable;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Test linear regression.
 * 
 */
public class TestLinearRegression {

  @Test
  public void testReadWriteLinearRegression() {
    String modelPath = "/tmp/testWriteReadLinearRegression.data";
    double learningRate = 0.2;
    double regularization = 0.1; // no regularization
    int dimension = 5;
    LinearRegression model = new LinearRegression(dimension);
    model.setLearningRate(learningRate);
    model.setRegularizationWeight(regularization);
    DoubleMatrix weights = model.getWeightsByLayer(0);
    try {
      model.writeModelToFile(modelPath);
    } catch (IOException e) {
      e.printStackTrace();
    }

    try {
      // read the meta-data
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(conf);
      model = new LinearRegression(modelPath);
      assertEquals(learningRate, model.getLearningRate(), 0.001);
      assertEquals(regularization, model.getRegularizationWeight(), 0.001);
      DoubleMatrix readWeights = model.getWeightsByLayer(0);
      assertEquals(weights.getRowCount(), readWeights.getRowCount());
      assertEquals(weights.getColumnCount(), readWeights.getColumnCount());
      assertEquals(dimension + 1, model.getLayerSize(0));
      for (int i = 0; i < weights.getRowCount(); ++i) {
        assertArrayEquals(weights.getRowVector(i).toArray(), readWeights
            .getRowVector(i).toArray(), 0.001);
      }
      // delete test file
      fs.delete(new Path(modelPath), true);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Ignore
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
    }

    // within 5% of error
    DoubleVector instance = new DenseDoubleVector(new double[] { 10, 10, 10 });
    assertEquals(13, regression.getOutput(instance).get(0), 13 * 0.05);
  }

  @Ignore
  @Test
  public void testDistributedTraining() {
    // write some data to input path
    Configuration conf = new Configuration();
    String strDataPath = "/tmp/sampleModel-testWriteReadMLP.data";
    Path dataPath = new Path(strDataPath);
    Random rnd = new Random();
    // y = 0.3 * x1 + 0.5 * x2 + 0.5 * x3
    DoubleVector[] trainingInstance = new DoubleVector[] {
        new DenseDoubleVector(new double[] { 1, 1, 1,
            1.3 + rnd.nextDouble() / 10 }),
        new DenseDoubleVector(new double[] { 2, 3, 5,
            4.6 + rnd.nextDouble() / 10 }),
        new DenseDoubleVector(new double[] { 3, 5, 2,
            4.4 + rnd.nextDouble() / 10 }),
        new DenseDoubleVector(
            new double[] { 5, 2, 1, 3 + rnd.nextDouble() / 10 }),
        new DenseDoubleVector(new double[] { 3, 1, 2,
            2.4 + rnd.nextDouble() / 10 }) };

    FileSystem fs = null;
    try {
      URI uri = new URI(strDataPath);
      fs = FileSystem.get(uri, conf);
      fs.delete(dataPath, true); // delete if exists
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, dataPath,
          LongWritable.class, VectorWritable.class);

      for (int i = 0; i < 1000; ++i) {
        VectorWritable vecWritable = new VectorWritable(trainingInstance[i % 4]);
        writer.append(new LongWritable(i), vecWritable);
      }
      writer.close();

      // train model
      int dimension = 3;
      LinearRegression regression = new LinearRegression(dimension);
      Map<String, String> trainingParams = new HashMap<String, String>();
      // initialize training parameter
      trainingParams.put("training.mode", "minibatch");
      trainingParams.put("training.batchsize", "" + 1000);
      regression.train(dataPath, trainingParams);

      // delete data
      fs.delete(dataPath, true);

    } catch (URISyntaxException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (InterruptedException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

  }

}
