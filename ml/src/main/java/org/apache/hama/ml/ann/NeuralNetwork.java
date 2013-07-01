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

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hama.ml.math.DoubleDoubleFunction;
import org.apache.hama.ml.math.FunctionFactory;
import org.mortbay.log.Log;

/**
 * NeuralNetwork defines the general operations for all the derivative models.
 * Typically, all derivative models such as Linear Regression, Logistic
 * Regression, and Multilayer Perceptron consist of neurons and the weights
 * between neurons.
 * 
 */
abstract class NeuralNetwork implements Writable{

  protected double learningRate = 0.5;

  //    the name of the model
  protected String modelType;
  
  protected String modelPath;

  public NeuralNetwork() {
    this.setModelType();
  }
  
  protected void setLearningRate(double learningRate) {
    if (learningRate <= 0) {
      throw new IllegalArgumentException("Learning rate must larger than 0.");
    }
    this.learningRate = learningRate;
  }

  /**
   * Set the modelType variable to specify the model type.
   */
  protected abstract void setModelType();
  
  /**
   * Train the model with the path of given training data and parameters.
   * 
   * @param dataInputPath The path of the training data.
   * @param trainingParams The parameters for training.
   * @throws IOException
   */
  protected void train(Path dataInputPath, Map<String, String> trainingParams)
      throws IOException {
    trainInternal(dataInputPath, trainingParams);
    // reload learned model

    Log.info(String.format("Reload model from %s.",
        trainingParams.get("modelPath")));
    this.modelPath = trainingParams.get("modelPath");
    this.readFromModel();
  }

  /**
   * Train the model with the path of given training data and parameters.
   * 
   * @param dataInputPath
   * @param trainingParams
   */
  protected abstract void trainInternal(Path dataInputPath,
      Map<String, String> trainingParams);

  /**
   * Read the model meta-data from the specified location.
   * 
   * @throws IOException
   */
  protected void readFromModel() throws IOException {
    Configuration conf = new Configuration();
    try {
      URI uri = new URI(modelPath);
      FileSystem fs = FileSystem.get(uri, conf);
      FSDataInputStream is = new FSDataInputStream(fs.open(new Path(modelPath)));
      this.readFields(is);
      if (!this.modelType.equals(this.getClass().getName())) {
        throw new IllegalStateException(String.format(
            "Model type incorrect, cannot load model '%s' for '%s'.",
            this.modelType, this.getClass().getName()));
      }
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }
  }

  /**
   * Write the model data to specified location.
   * 
   * @param modelPath The location in file system to store the model.
   * @throws IOException
   */
  public void writeModelToFile(String modelPath) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    FSDataOutputStream stream = fs.create(new Path(modelPath), true);
    this.write(stream);
    stream.close();
  }

}
