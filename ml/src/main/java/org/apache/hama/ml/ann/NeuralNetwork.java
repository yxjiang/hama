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

<<<<<<< HEAD
=======
import java.io.DataInput;
import java.io.DataOutput;
>>>>>>> upstream/trunk
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
<<<<<<< HEAD
import org.apache.hama.ml.math.DoubleDoubleFunction;
import org.apache.hama.ml.math.FunctionFactory;
import org.mortbay.log.Log;
=======
import org.apache.hadoop.io.WritableUtils;
>>>>>>> upstream/trunk

import com.google.common.base.Preconditions;

/**
 * NeuralNetwork defines the general operations for all the derivative models.
 * Typically, all derivative models such as Linear Regression, Logistic
 * Regression, and Multilayer Perceptron consist of neurons and the weights
 * between neurons.
 * 
 */
abstract class NeuralNetwork implements Writable {

<<<<<<< HEAD
  public static final double DEFAULT_LEARNING_RATE = 0.5;
  
  protected double learningRate = DEFAULT_LEARNING_RATE;

  // the name of the model
  protected String modelType;

  protected String modelPath;

  public NeuralNetwork() {
    this.setModelType();
=======
  private static final double DEFAULT_LEARNING_RATE = 0.5;

  protected double learningRate;
  protected boolean learningRateDecay = false;

  // the name of the model
  protected String modelType;
  // the path to store the model
  protected String modelPath;

  public NeuralNetwork() {
    this.learningRate = DEFAULT_LEARNING_RATE;
    this.modelType = this.getClass().getSimpleName();
>>>>>>> upstream/trunk
  }

  public NeuralNetwork(String modelPath) {
    try {
      this.modelPath = modelPath;
      this.readFromModel();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * Set the degree of aggression during model training, a large learning rate
   * can increase the training speed, but it also decrease the chance of model
   * converge. Recommend in range (0, 0.3).
   * 
   * @param learningRate
   */
  public void setLearningRate(double learningRate) {
    Preconditions.checkArgument(learningRate > 0,
        "Learning rate must larger than 0.");
    this.learningRate = learningRate;
  }

  public double getLearningRate() {
    return this.learningRate;
  }

<<<<<<< HEAD
  /**
   * Set the modelType variable to specify the model type.
   */
  protected abstract void setModelType();
=======
  public void isLearningRateDecay(boolean decay) {
    this.learningRateDecay = decay;
  }

  public String getModelType() {
    return this.modelType;
  }
>>>>>>> upstream/trunk

  /**
   * Train the model with the path of given training data and parameters.
   * 
   * @param dataInputPath The path of the training data.
   * @param trainingParams The parameters for training.
   * @throws IOException
   */
<<<<<<< HEAD
  public void train(Path dataInputPath, Map<String, String> trainingParams)
      throws IOException, InterruptedException, ClassNotFoundException {
    // set model path
    trainingParams.put("model.path", this.modelPath);
    // train with BSP job
    trainInternal(dataInputPath, trainingParams);
    // reload learned model
    Log.info(String.format("Reload model from %s.",
        trainingParams.get("modelPath")));
    this.modelPath = trainingParams.get("modelPath");
    this.readFromModel();
=======
  public void train(Path dataInputPath, Map<String, String> trainingParams) {
    Preconditions.checkArgument(this.modelPath != null,
        "Please set the model path before training.");
    // train with BSP job
    try {
      trainInternal(dataInputPath, trainingParams);
      // write the trained model back to model path
      this.readFromModel();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (InterruptedException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
>>>>>>> upstream/trunk
  }

  /**
   * Train the model with the path of given training data and parameters.
   * 
   * @param dataInputPath
   * @param trainingParams
   */
  protected abstract void trainInternal(Path dataInputPath,
      Map<String, String> trainingParams) throws IOException,
      InterruptedException, ClassNotFoundException;

  /**
   * Read the model meta-data from the specified location.
   * 
   * @throws IOException
   */
  protected void readFromModel() throws IOException {
<<<<<<< HEAD
    Configuration conf = new Configuration();
    try {
      URI uri = new URI(modelPath);
      FileSystem fs = FileSystem.get(uri, conf);
      FSDataInputStream is = new FSDataInputStream(fs.open(new Path(modelPath)));
      this.readFields(is);
      Preconditions.checkArgument(this.modelType.equals(this.getClass()
          .getSimpleName()), String.format(
          "Model type incorrect, cannot load model '%s' for '%s'.",
          this.modelType, this.getClass().getSimpleName()));
=======
    Preconditions.checkArgument(this.modelPath != null,
        "Model path has not been set.");
    Configuration conf = new Configuration();
    try {
      URI uri = new URI(this.modelPath);
      FileSystem fs = FileSystem.get(uri, conf);
      FSDataInputStream is = new FSDataInputStream(fs.open(new Path(modelPath)));
      this.readFields(is);
>>>>>>> upstream/trunk
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
<<<<<<< HEAD
  public void writeModelToFile(String modelPath) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    FSDataOutputStream stream = fs.create(new Path(modelPath), true);
    this.write(stream);
    stream.close();
  }
  
  /**
   * Set the model path.
=======
  public void writeModelToFile() throws IOException {
    Preconditions.checkArgument(this.modelPath != null,
        "Model path has not been set.");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    FSDataOutputStream stream = fs.create(new Path(this.modelPath), true);
    this.write(stream);
    stream.close();
  }

  /**
   * Set the model path.
   * 
>>>>>>> upstream/trunk
   * @param modelPath
   */
  public void setModelPath(String modelPath) {
    this.modelPath = modelPath;
  }
<<<<<<< HEAD
  
  /**
   * Get the model path.
=======

  /**
   * Get the model path.
   * 
>>>>>>> upstream/trunk
   * @return
   */
  public String getModelPath() {
    return this.modelPath;
  }

<<<<<<< HEAD
=======
  public void readFields(DataInput input) throws IOException {
    // read model type
    this.modelType = WritableUtils.readString(input);
    // read learning rate
    this.learningRate = input.readDouble();
    // read model path
    this.modelPath = WritableUtils.readString(input);
    if (this.modelPath.equals("null")) {
      this.modelPath = null;
    }
  }

  public void write(DataOutput output) throws IOException {
    // write model type
    WritableUtils.writeString(output, modelType);
    // write learning rate
    output.writeDouble(learningRate);
    // write model path
    if (this.modelPath != null) {
      WritableUtils.writeString(output, modelPath);
    } else {
      WritableUtils.writeString(output, "null");
    }
  }

>>>>>>> upstream/trunk
}
