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
package org.apache.hama.ml.perception;

/**
 * The logistic cost function.
 * 
 * <pre>
 * cost(t, y) = - t * log(y) - (1 - t) * log(1 - y),
 * where t denotes the target value, y denotes the estimated value.
 * </pre>
 */
public class LogisticCostFunction extends CostFunction {

  @Override
  public double getCost(double target, double actual) {
    return -target * Math.log(actual) - (1 - target) * Math.log(1 - actual);
  }

  @Override
  public double getPartialDerivative(double target, double actual) {
    if (actual == 1) {
      actual = 0.999;
    } else if (actual == 0) {
      actual = 0.001;
    }
    if (target == 1) {
      target = 0.999;
    } else if (target == 0) {
      target = 0.001;
    }
    return -target / actual + (1 - target) / (1 - actual);
  }

}
