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
import org.apache.hama.ml.math.FunctionFactory;

/**
 * NeuralNetwork defines the general operations for all the derivative models.
 * Typically, all derivative models such as Linear Regression, Logistic
 * Regression, and Multilayer Perceptron consist of neurons and the weights
 * between neurons.
 * 
 */
public abstract class NeuralNetwork {

  protected double learningRate = 0.5;

  /* The cost function of the model */
  protected DoubleDoubleFunction costFunction;

  protected void setLearningRate(double learningRate) {
    if (learningRate <= 0) {
      throw new IllegalArgumentException("Learning rate must larger than 0.");
    }
    this.learningRate = learningRate;
  }

  /**
   * Set the cost function for the model.
   * 
   * @param costFunctionName
   */
  protected void setCostFunction(String costFunctionName) {
    this.costFunction = FunctionFactory
        .createDoubleDoubleFunction(costFunctionName);
  }

}
