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
package org.apache.hama.ml.math;

import java.lang.reflect.Field;




/**
 * DoubleDoubleFunctions contains the commonly used DoubleDoubleFunction.
 * 
 */
public class DoubleDoubleFunctions {
  
  /**
   * Square error cost function.
   * 
   * <pre>
   * cost(x1, x2) = 0.5 * (x1 - x2) &circ; 2
   * </pre>
   */
  public final static DoubleDoubleFunction SQUARED_ERROR = new DoubleDoubleFunction() {

    @Override
    public double calculate(double target, double actual) {
      double diff = target - actual;
      return 0.5 * diff * diff;
    }

    @Override
    public double calculateDerivative(double target, double actual) {
      return target - actual;
    }

  };

  /**
   * The cross entropy cost function.
   * 
   * <pre>
   * cost(t, y) = - t * log(y) - (1 - t) * log(1 - y),
   * where t denotes the target value, y denotes the estimated value.
   * </pre>
   */
  public final static DoubleDoubleFunction CROSS_ENTROPY = new DoubleDoubleFunction() {

    @Override
    public double calculate(double target, double actual) {
      return -target * Math.log(actual) - (1 - target) * Math.log(1 - actual);
    }

    @Override
    public double calculateDerivative(double target, double actual) {
      double adjustedTarget = target;
      double adjustedActual = actual;
      if (adjustedActual == 1) {
        adjustedActual = 0.999;
      } else if (actual == 0) {
        adjustedActual = 0.001;
      }
      if (adjustedTarget == 1) {
        adjustedTarget = 0.999;
      } else if (adjustedTarget == 0) {
        adjustedTarget = 0.001;
      }
      return -adjustedTarget / adjustedActual + (1 - adjustedTarget)
          / (1 - adjustedActual);
    }

  };
  
}
