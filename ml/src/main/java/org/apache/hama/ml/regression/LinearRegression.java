package org.apache.hama.ml.regression;

import org.apache.hama.ml.ann.SmallLayeredNeuralNetwork;

/**
 * Linear regression model. The training can be conducted in parallel, but the
 * weights are assumed to be stored in a single machine.
 * 
 */
public class LinearRegression extends SmallLayeredNeuralNetwork {
  
  /**
   * Initialize the linear regression model.
   * @param dimension
   */
  public LinearRegression(int dimension) {
    
  }

}
