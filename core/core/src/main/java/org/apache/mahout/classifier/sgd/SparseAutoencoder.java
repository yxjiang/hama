/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleDoubleFunction;

public class SparseAutoencoder extends MultiLayerPerceptron{
  
  
  /* Sparse Autoencoder */
  
  /**
   * Constructor for MLP Sparse Autoencoder
   * @param nbInputOutputUnits
   */
  public SparseAutoencoder(int nbInputOutputUnits) {
    super(nbInputOutputUnits, nbInputOutputUnits, new int[]{nbInputOutputUnits}, false);  
    average_activation = new DenseVector(nbUnits[1]);
  }
  
  private Vector average_activation = null;
  private double estimationUpdateWeight = 0.0001;
  private double targetActivation = -0.98;
  private double sparseLearningRate = 0.5;
  
  @Override
  public Vector trainOnline(Vector input, Vector target) {
    
    super.trainOnline(input, target);
    
    enforceSparseConstraint(units[1], weights[0]);
    
    return units[nbLayer - 1];
  }
  
  private void enforceSparseConstraint(Vector activation, Matrix weights){
    updateAverageActivation(activation);
    updateBiasWeights(weights);
  }
  
  private void updateAverageActivation(Vector activation){
    average_activation.assign(activation, new DoubleDoubleFunction() {
      @Override
      public double apply(double a, double b) {
        return (1d - estimationUpdateWeight) * a + estimationUpdateWeight * b;
      }; 
    });
  }
  
  private void updateBiasWeights(Matrix weights){
    int nbUnit = weights.rowSize();
    // init with i=1: not for weights to bias! 
    for(int i=1; i<nbUnit; i++){
      weights.set(i, 0 ,
          weights.get(i, 0) - learningRate * sparseLearningRate *
          (average_activation.getQuick(i) - targetActivation) );
    }
  }
  
  /**
   * Chainable configuration option.
   * 
   * @param updateWeight 
   * @return This, so other configurations can be chained.
   */
  public SparseAutoencoder estimationUpdateWeight(double estimationUpdateWeight) {
    this.estimationUpdateWeight = estimationUpdateWeight;
    return this;
  }

  /**
   * Chainable configuration option.
   * 
   * @param targetActivation 
   * @return This, so other configurations can be chained.
   */
  public SparseAutoencoder targetActivation(double targetActivation){
    this.targetActivation = targetActivation;
    return this;
  }
  
  /**
   * Chainable configuration option.
   * 
   * @param sparseLearningRate
   * @return This, so other configurations can be chained.
   */
  public SparseAutoencoder sparseLearningRate(double sparseLearningRate){
    this.sparseLearningRate = sparseLearningRate;
    return this;
  }
 
}
