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
<<<<<<< HEAD
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
=======

package org.apache.hama.ml.regression;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleVector;
import org.junit.Test;
import org.mortbay.log.Log;

/**
 * Test the functionalities of the linear regression model.
>>>>>>> upstream/trunk
 * 
 */
public class TestLinearRegression {

  @Test
<<<<<<< HEAD
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

=======
  public void testLinearRegressionSimple() {
    // y = 2.1 * x_1 + 0.7 * x_2 * 0.1 * x_3
    double[][] instances = { { 1, 1, 1, 2.9 }, { 5, 2, 3, 12.2 },
        { 2, 5, 8, 8.5 }, { 0.5, 0.1, 0.2, 1.14 }, { 10, 20, 30, 38 },
        { 0.6, 20, 5, 16.76 } };

    LinearRegression regression = new LinearRegression(instances[0].length - 1);
    regression.setLearningRate(0.001);
    regression.setMomemtumWeight(0.1);

    int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
      for (int j = 0; j < instances.length; ++j) {
        regression.trainOnline(new DenseDoubleVector(instances[j]));
      }
    }

    double relativeError = 0;
    for (int i = 0; i < instances.length; ++i) {
      DoubleVector test = new DenseDoubleVector(instances[i]);
      double expected = test.get(test.getDimension() - 1);
      test = test.slice(test.getDimension() - 1);
      double actual = regression.getOutput(test).get(0);
      relativeError += Math.abs((expected - actual) / expected);
    }

    relativeError /= instances.length;
    Log.info(String.format("Relative error %f%%\n", relativeError));
  }

  @Test
  public void testLinearRegressionOnlineTraining() {
    // read linear regression data
    String filepath = "src/test/resources/linear_regression_data.txt";
    List<double[]> instanceList = new ArrayList<double[]>();

    try {
      BufferedReader br = new BufferedReader(new FileReader(filepath));
      String line = null;
      while ((line = br.readLine()) != null) {
        if (line.startsWith("#")) { // ignore comments
          continue;
        }
        String[] tokens = line.trim().split(" ");
        double[] instance = new double[tokens.length];
        for (int i = 0; i < tokens.length; ++i) {
          instance[i] = Double.parseDouble(tokens[i]);
        }
        instanceList.add(instance);
      }
      br.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
    // divide dataset into training and testing
    List<double[]> testInstances = new ArrayList<double[]>();
    testInstances.addAll(instanceList.subList(instanceList.size() - 20,
        instanceList.size()));
    List<double[]> trainingInstances = instanceList.subList(0,
        instanceList.size() - 20);

    int dimension = instanceList.get(0).length - 1;

    LinearRegression regression = new LinearRegression(dimension);
    regression.setLearningRate(0.00000005);
    regression.setMomemtumWeight(0.1);
    regression.setRegularizationWeight(0.05);
    int iterations = 2000;
    for (int i = 0; i < iterations; ++i) {
      for (double[] trainingInstance : trainingInstances) {
        regression.trainOnline(new DenseDoubleVector(trainingInstance));
      }
    }

    double relativeError = 0.0;
    // calculate the error on test instance
    for (double[] testInstance : testInstances) {
      DoubleVector instance = new DenseDoubleVector(testInstance);
      double expected = instance.get(instance.getDimension() - 1);
      instance = instance.slice(instance.getDimension() - 1);
      double actual = regression.getOutput(instance).get(0);
      if (expected == 0) {
        expected = 0.0000001;
      }
      relativeError += Math.abs((expected - actual) / expected);
    }
    relativeError /= testInstances.size();

    Log.info(String.format("Relative error: %f%%\n", relativeError * 100));
>>>>>>> upstream/trunk
  }

}
