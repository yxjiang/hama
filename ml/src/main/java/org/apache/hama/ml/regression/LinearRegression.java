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
    
    // squared error by default
    this.costFunction = FunctionFactory
        .createDoubleDoubleFunction("SquaredError");
  }

  @Override
  protected void train(Path dataInputPath, Map<String, String> trainingParams) {
    // TODO Auto-generated method stub
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

}
