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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DoubleDoubleFunction;
import org.apache.hama.ml.math.DoubleFunction;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.FunctionFactory;

/**
 * SmallLayeredNeuralNetwork defines the general operations for derivative
 * layered models, include Linear Regression, Logistic Regression, Multilayer
 * Perceptron, Autoencoder, and Restricted Boltzmann Machine, etc. For
 * SmallLayeredNeuralNetwork, the training can be conducted in parallel, but the
 * parameters of the models are assumes to be stored in a single machine.
 * 
 * In general, these models consist of neurons which are aligned in layers.
 * Between layers, for any two adjacent layers, the neurons are connected to
 * form a bipartite weighted graph.
 * 
 */
public abstract class SmallLayeredNeuralNetwork extends
    AbstractLayeredNeuralNetwork {

  /* Record the size of each layer */
  protected List<Integer> layerSizeList;

  /* Weights between neurons at adjacent layers */
  protected List<DoubleMatrix> weightMatrixList;

  /* Different layers can have different squashing function */
  protected List<DoubleFunction> squashingFunctionList;

  public SmallLayeredNeuralNetwork() {
    this.layerSizeList = new ArrayList<Integer>();
    this.weightMatrixList = new ArrayList<DoubleMatrix>();
    this.squashingFunctionList = new ArrayList<DoubleFunction>();
  }

  @Override
  /**
   * {@inheritDoc}
   */
  protected int addLayer(int size, boolean isFinalLayer) {
    if (size <= 0) {
      throw new IllegalArgumentException("Size of layer must larger than 0.");
    }
    if (isFinalLayer) {
      size += 1;
    }

    this.layerSizeList.add(size);
    int layerIdx = this.layerSizeList.size() - 1;

    if (layerIdx > 0) { // add weights between current layer and previous layer
      int sizePrevLayer = this.layerSizeList.get(layerIdx - 1);
      DoubleMatrix weightMatrix = new DenseDoubleMatrix(sizePrevLayer, size);
      // initialize weights
      final Random rnd = new Random();
      weightMatrix.applyToElements(new DoubleFunction() {

        @Override
        public double apply(double value) {
          return rnd.nextDouble() - 0.5;
        }

        @Override
        public double applyDerivative(double value) {
          throw new UnsupportedOperationException("");
        }

      });
    }
    return layerIdx;
  }

  @Override
  /**
   * {@inheritDoc}
   */
  protected void setSquashingFunction(int layerIdx, String squashingFunctionName) {
    this.squashingFunctionList.set(layerIdx,
        FunctionFactory.createDoubleFunction(squashingFunctionName));
  }
  
}
