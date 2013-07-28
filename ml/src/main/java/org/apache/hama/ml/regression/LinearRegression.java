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

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hama.ml.ann.SmallLayeredNeuralNetwork;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.math.FunctionFactory;

/**
 * 
 *
 */
public class LinearRegression {

  /* Internal model */
  private SmallLayeredNeuralNetwork ann;

  public LinearRegression(int dimension) {
    ann = new SmallLayeredNeuralNetwork();
    ann.addLayer(dimension, false,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.addLayer(1, true,
        FunctionFactory.createDoubleFunction("IdentityFunction"));
    ann.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction("SquaredError"));
  }

  public LinearRegression(String modelPath) {
    ann = new SmallLayeredNeuralNetwork(modelPath);
  }

  public void setLearningRate(double learningRate) {
    ann.setLearningRate(learningRate);
  }

  /**
   * Train the linear regression model with one instance.
   * 
   * @param trainingInstance
   */
  public void trainOnline(DoubleVector trainingInstance) {
    ann.trainOnline(trainingInstance);
  }

  /**
   * Train the model with given data.
   * 
   * @param dataInputPath The file path that contains the training instance.
   * @param trainingParams The training parameters.
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void train(Path dataInputPath, Map<String, String> trainingParams) {
    try {
      ann.train(dataInputPath, trainingParams);
    } catch (IOException e) {
      e.printStackTrace();
    } catch (InterruptedException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
  }

  /**
   * Get the output according to given input instance.
   * 
   * @param instance
   * @return
   */
  public DoubleVector getOutput(DoubleVector instance) {
    return ann.getOutput(instance);
  }

  /**
   * Set the path to store the model. Note this is just set the path, it does
   * not save the model. You should call writeModelToFile to save the model.
   * 
   * @param modelPath
   */
  public void setModelPath(String modelPath) {
    ann.setModelPath(modelPath);
  }

  /**
   * Save the model to specified model path.
   */
  public void writeModelToFile() {
    try {
      ann.writeModelToFile();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  /**
   * Get the weights of the model.
   * @return
   */
  public DoubleVector getWeights() {
    return ann.getWeightsByLayer(0).getRowVector(0);
  }

}
