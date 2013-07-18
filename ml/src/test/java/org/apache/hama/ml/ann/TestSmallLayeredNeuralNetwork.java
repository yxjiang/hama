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

import static org.junit.Assert.assertEquals;

import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.math.FunctionFactory;
import org.junit.Test;

/**
 * Test the functionality of SmallLayeredNeuralNetwork.
 * 
 */
public class TestSmallLayeredNeuralNetwork {
  
  @Test
  public void testOutput() {
    
    SmallLayeredNeuralNetwork ann = new SmallLayeredNeuralNetwork();
    ann.addLayer(2, false);
    ann.addLayer(5, false);
    ann.addLayer(1, true);
    ann.setCostFunction(FunctionFactory.createDoubleDoubleFunction("SquaredError"));
    ann.setLearningRate(0.1);
    ann.setSquashingFunction(FunctionFactory.createDoubleFunction("IdentityFunction"));
    double[] arr = new double[] {0, 1};
    DoubleVector training = new DenseDoubleVector(arr);
    DoubleVector result = ann.getOutput(training);
    assertEquals(1, result.getDimension());
    assertEquals(3, result.get(0), 0.000001);
  }

}
