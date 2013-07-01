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

import org.apache.hama.ml.math.DoubleDoubleFunction;
import org.apache.hama.ml.math.DoubleFunction;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.FunctionFactory;

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

  /* The cost function of the model */
  protected DoubleDoubleFunction costFunction;
  
  /**
   * Set the cost function for the model.
   * 
   * @param costFunctionName
   */
  protected void setCostFunction(String costFunctionName) {
    this.costFunction = FunctionFactory
        .createDoubleDoubleFunction(costFunctionName);
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
   * Set the squashing function of the specified layer. It will have no effect
   * if the specified layer is the input layer.
   * 
   * @param layerIdx
   * @param squashingFunction The  the squashing function.
   */
  protected abstract void setSquashingFunction(int layerIdx,
      DoubleFunction squashingFunction);
  
  /**
   * Set the squashing function for all layers.
   * @param squashingFunction
   */
  protected abstract void setSquashingFunction(DoubleFunction squashingFunction);

  /**
   * Get the weights between layer layerIdx and layerIdx + 1
   * 
   * @param layerIdx The index of the layer
   * @return The weights in form of {@link DoubleMatrix}
   */
  public abstract DoubleMatrix getWeightsByLayer(int layerIdx);

}
