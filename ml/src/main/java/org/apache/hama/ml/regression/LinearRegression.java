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
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.FunctionFactory;

/**
 * Linear regression model. The training can be conducted in parallel, but the
 * weights are assumed to be stored in a single machine.
 * 
 */
public class LinearRegression extends SmallLayeredNeuralNetwork {

  /**
   * Initialize the linear regression model.
   * 
   * @param dimension
   */
  public LinearRegression(int dimension) {
    // initialize topology
    this.addLayer(dimension, false);
    this.addLayer(1, true);
    this.learningRate = 1.0;
    
    // squared error by default
    this.costFunction = FunctionFactory
        .createDoubleDoubleFunction("SquaredError");
    this.setSquashingFunction(FunctionFactory.createDoubleFunction("Identity"));
  }

  @Override
  public DoubleMatrix getWeightsByLayer(int layerIdx) {
    if (layerIdx < 0 || layerIdx >= this.weightMatrixList.size()) {
      throw new IllegalArgumentException("Invalid layer index.");
    }
    return this.weightMatrixList.get(layerIdx);
  }

  @Override
  protected void readFromModel() throws IOException {
    // TODO Auto-generated method stub
    
  }

  @Override
  public void writeModelToFile(String modelPath) throws IOException {
    // TODO Auto-generated method stub
    
  }

  @Override
  protected void trainInternal(Path dataInputPath,
      Map<String, String> trainingParams) {
    // TODO Auto-generated method stub
    
  }

}
