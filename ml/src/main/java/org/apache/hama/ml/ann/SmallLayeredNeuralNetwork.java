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
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleFunction;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;

import com.google.common.base.Preconditions;

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

  /* Weights between neurons at adjacent layers */
  protected List<DoubleMatrix> weightMatrixList;

  /* Different layers can have different squashing function */
  protected List<DoubleFunction> squashingFunctionList;

  public SmallLayeredNeuralNetwork() {
    this.layerSizeList = new ArrayList<Integer>();
    this.weightMatrixList = new ArrayList<DoubleMatrix>();
    this.squashingFunctionList = new ArrayList<DoubleFunction>();
  }

  public SmallLayeredNeuralNetwork(String modelPath) {
    super(modelPath);
  }

  @Override
  /**
   * {@inheritDoc}
   */
  protected int addLayer(int size, boolean isFinalLayer) {
    Preconditions.checkArgument(size > 0, "Size of layer must larger than 0.");
    if (!isFinalLayer) {
      size += 1;
    }

    this.layerSizeList.add(size);
    int layerIdx = this.layerSizeList.size() - 1;

    if (layerIdx > 0) { // add weights between current layer and previous layer
      int sizePrevLayer = this.layerSizeList.get(layerIdx - 1);
      // row count equals to size of current size and column count equals to
      // size of previous layer
      DoubleMatrix weightMatrix = new DenseDoubleMatrix(size, sizePrevLayer);
      // initialize weights
      final Random rnd = new Random();
      weightMatrix.applyToElements(new DoubleFunction() {

        @Override
        public double apply(double value) {
          //
          return 0.5;
        }

        @Override
        public double applyDerivative(double value) {
          throw new UnsupportedOperationException("");
        }

      });
      this.weightMatrixList.add(weightMatrix);
      this.squashingFunctionList.add(null);
    }
    return layerIdx;
  }

  @Override
  /**
   * {@inheritDoc}
   */
  public void setSquashingFunction(int layerIdx,
      DoubleFunction squashingFunction) {
    this.squashingFunctionList.set(layerIdx, squashingFunction);
  }

  /**
   * {@inheritDoc}
   */
  public void setSquashingFunction(DoubleFunction squashingFunction) {
    for (int i = 0; i < squashingFunctionList.size(); ++i) {
      this.setSquashingFunction(i, squashingFunction);
    }
  }

  /**
   * Update the weight matrices with given matrices.
   * 
   * @param matrices
   */
  public void updateWeightMatrices(DoubleMatrix[] matrices) {
    for (int i = 0; i < matrices.length; ++i) {
      DoubleMatrix matrix = this.weightMatrixList.get(i);
      this.weightMatrixList.set(i, matrix.add(matrices[i]));
    }
  }

  /**
   * Add a batch of matrices onto the given destination matrices.
   * 
   * @param destMatrices
   * @param sourceMatrices
   */
  public static void matricesAdd(DoubleMatrix[] destMatrices,
      DoubleMatrix[] sourceMatrices) {
    Preconditions
        .checkArgument(
            destMatrices != null && sourceMatrices != null
                && destMatrices.length == sourceMatrices.length,
            "Number of matrices should be equal for both destination matrices and source matrices.");
    for (int i = 0; i < destMatrices.length; ++i) {
      destMatrices[i] = destMatrices[i].add(sourceMatrices[i]);
    }
  }
  
  /**
   * Get all the weight matrices.
   * @return
   */
  public DoubleMatrix[] getWeightMatrices() {
    return (DoubleMatrix[])this.weightMatrixList.toArray();
  }

  /**
   * Get the output of the model according to given feature instance.
   */
  public DoubleVector getOutput(DoubleVector instance) {
    // add bias feature
    DoubleVector instanceWithBias = new DenseDoubleVector(
        instance.getDimension() + 1);
    instanceWithBias.set(0, 1);
    for (int i = 1; i < instanceWithBias.getDimension(); ++i) {
      instanceWithBias.set(i, instance.get(i - 1));
    }

    List<DoubleVector> outputCache = getOutputInternal(instanceWithBias);
    // return the output of the last layer
    return outputCache.get(outputCache.size() - 1);
  }

  /**
   * Calculate output internally, the intermediate output of each layer will be
   * stored.
   * 
   * @param instance The instance contains the features.
   * @return Cached output of each layer.
   */
  public List<DoubleVector> getOutputInternal(DoubleVector instance) {
    List<DoubleVector> outputCache = new ArrayList<DoubleVector>();
    // fill with instance
    DoubleVector intermediateOutput = instance;
    outputCache.add(intermediateOutput);
    // System.out.println("Input:");
    // for (int j = 0; j < intermediateOutput.getDimension(); ++j) {
    // System.out.printf("%f ", intermediateOutput.get(j));
    // }
    // System.out.println();

    for (int i = 0; i < this.layerSizeList.size() - 1; ++i) {
      intermediateOutput = forward(i, intermediateOutput);
      // System.out.println("For layer i + 1");
      for (int j = 0; j < intermediateOutput.getDimension(); ++j) {
        // System.out.printf("%f ", intermediateOutput.get(j));
      }
      System.out.println();
      outputCache.add(intermediateOutput);
    }
    return outputCache;
  }

  /**
   * Forward the calculation for one layer.
   * 
   * @param fromLayer The index of the previous layer.
   * @param intermediateOutput The intermediateOutput of previous layer.
   * @return
   */
  protected DoubleVector forward(int fromLayer, DoubleVector intermediateOutput) {
    return this.weightMatrixList.get(fromLayer)
        .multiplyVectorUnsafe(intermediateOutput)
        .applyToElements(this.squashingFunctionList.get(fromLayer));
  }

}
