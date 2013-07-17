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

import java.util.List;

import org.apache.hama.ml.math.DoubleDoubleFunction;
import org.apache.hama.ml.math.DoubleFunction;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.math.FunctionFactory;

import com.google.common.base.Preconditions;

/**
 * AbstractLayeredNeuralNetwork defines the general operations for derivative
 * layered models, include Linear Regression, Logistic Regression, Multilayer
 * Perceptron, Autoencoder, and Restricted Boltzmann Machine, etc.
 * 
 * In general, these models consist of neurons which are aligned in layers.
 * Between layers, for any two adjacent layers, the neurons are connected to
 * form a bipartite weighted graph.
 * 
 */
abstract class AbstractLayeredNeuralNetwork extends NeuralNetwork {

  public static final double DEFAULT_REGULARIZATION_WEIGHT = 0;
  /* Record the size of each layer */
  protected List<Integer> layerSizeList;

  /* The weight of regularization */
  protected double regularizationWeight = DEFAULT_REGULARIZATION_WEIGHT;

  /* The cost function of the model */
  protected DoubleDoubleFunction costFunction;
  
  public static enum TrainingMethod {
    GRADIATE_DESCENT
  }

  public AbstractLayeredNeuralNetwork() {
  }

  public AbstractLayeredNeuralNetwork(String modelPath) {
    super(modelPath);
  }

  /**
   * Set the regularization weight. Recommend in the range [0, 0.5), More
   * complex the model is, less weight the regularization is.
   * 
   * @param regularization
   */
  public void setRegularizationWeight(double regularizationWeight) {
    Preconditions.checkArgument(regularizationWeight >= 0
        && regularizationWeight < 1,
        "Regularization weight must be in range [0, 1.)");
    this.regularizationWeight = regularizationWeight;
  }

  public double getRegularizationWeight() {
    return this.regularizationWeight;
  }

  /**
   * Set the cost function for the model.
   * 
   * @param costFunctionName
   */
  public void setCostFunction(DoubleDoubleFunction costFunction) {
    this.costFunction = costFunction;
  }

  /**
   * Add a layer of neurons with specified size. If the added layer is not the
   * first layer, it will automatically connects the neurons between with the
   * previous layer.
   * 
   * @param size
   * @param isFinalLayer If false, add a bias neuron.
   * @return The layer index, starts with 0.
   */
  protected abstract int addLayer(int size, boolean isFinalLayer);

  /**
   * Get the size of a particular layer.
   * 
   * @param layer
   * @return
   */
  public int getLayerSize(int layer) {
    Preconditions.checkArgument(
        layer >= 0 && layer < this.layerSizeList.size() - 1,
        String.format("Input must be in range [0, %d]\n",
            this.layerSizeList.size() - 1));
    return this.layerSizeList.get(layer);
  }

  /**
   * Get the layer size list.
   * 
   * @return
   */
  List<Integer> getLayerSizeList() {
    return this.layerSizeList;
  }

  /**
   * Set the squashing function of the specified layer. It will have no effect
   * if the specified layer is the output layer.
   * 
   * @param layerIdx
   * @param squashingFunction The the squashing function.
   */
  protected abstract void setSquashingFunction(int layerIdx,
      DoubleFunction squashingFunction);

  /**
   * Set the squashing function for all layers.
   * 
   * @param squashingFunction
   */
  public abstract void setSquashingFunction(DoubleFunction squashingFunction);

  /**
   * Get the weights between layer layerIdx and layerIdx + 1
   * 
   * @param layerIdx The index of the layer
   * @return The weights in form of {@link DoubleMatrix}
   */
  public abstract DoubleMatrix getWeightsByLayer(int layerIdx);

  /**
   * Get the updated weights using one training instance.
   * 
   * @param trainingInstance The trainingInstance is the concatenation of
   *          feature vector and class label vector.
   * @return The update of each weight, in form of matrix list.
   * @throws Exception
   */
  public abstract DoubleMatrix[] trainByInstance(DoubleVector trainingInstance, TrainingMethod method);

  /**
   * Get the output calculated by the model.
   * 
   * @param instance The feature instance.
   * @return
   */
  public abstract DoubleVector getOutput(DoubleVector instance);

}
