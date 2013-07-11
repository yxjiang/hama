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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.ml.ann.SmallLayeredNeuralNetwork;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DenseDoubleVector;
import org.apache.hama.ml.math.DoubleDoubleFunction;
import org.apache.hama.ml.math.DoubleFunction;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.math.FunctionFactory;
import org.apache.hama.ml.writable.MatrixWritable;

import com.google.common.base.Preconditions;

/**
 * Linear regression model. The training can be conducted in parallel, but the
 * weights are assumed to be stored in a single machine.
 * 
 */
public final class LinearRegression extends SmallLayeredNeuralNetwork implements
    Writable {

  /**
   * Initialize the linear regression model.
   * 
   * @param dimension
   */
  public LinearRegression(int dimension) {
    // initialize topology
    this.addLayer(dimension, false);
    this.addLayer(1, true);
    this.learningRate = 0.5;
    this.regularization = 0;

    // squared error by default
    this.costFunction = FunctionFactory
        .createDoubleDoubleFunction("SquaredError");
    this.setSquashingFunction(FunctionFactory
        .createDoubleFunction("IdentityFunction"));
  }

  @Override
  public DoubleMatrix[] trainByInstance(DoubleVector trainingInstance) {
    int weightDimension = this.layerSizeList.get(0);
    // include bias, the dimension of feature equals to the dimension of
    // training instance
    Preconditions.checkArgument(
        trainingInstance != null
            && trainingInstance.getDimension() == weightDimension,
        "The dimension of training instance does not match the model.");

    DoubleVector featureVector = trainingInstance.slice(trainingInstance
        .getDimension());
    // insert bias
    featureVector.set(0, 1);
    for (int i = 1; i < featureVector.getDimension(); ++i) {
      featureVector.set(i, trainingInstance.get(i - 1));
    }

    final double expected = trainingInstance.get(trainingInstance
        .getDimension() - 1);
    int lastLayerIdx = 1;
    final double actual = this.getOutputInternal(featureVector)
        .get(lastLayerIdx).get(0);

    DoubleMatrix[] weightUpdateMatrices = new DoubleMatrix[1];
    // obtain matrix update by gradient descent
    weightUpdateMatrices[0] = new DenseDoubleMatrix(1, weightDimension);
    // update weights for each dimension
    for (int i = 0; i < weightDimension; ++i) {
      double delta = this.costFunction.applyDerivative(expected, actual)
          * featureVector.get(i);
      // partial derivative of regularization
      delta += this.regularization * 2 * this.weightMatrixList.get(0).get(0, i);
      weightUpdateMatrices[0].set(0, i, -this.learningRate * delta);
    }

    return weightUpdateMatrices;
  }

  /**
   * Set the cost function to evaluate the error made by linear regression.
   * 
   * @param costFunction The instance of cost function.
   */
  public void setCostFunction(DoubleDoubleFunction costFunction) {
    this.costFunction = costFunction;
  }

  @Override
  public DoubleMatrix getWeightsByLayer(int layerIdx) {
    Preconditions.checkArgument(layerIdx >= 0
        && layerIdx < this.weightMatrixList.size(),
        "Invalide layer index when obtaining weight matrix.");
    return this.weightMatrixList.get(layerIdx);
  }

  @Override
  protected void trainInternal(Path dataInputPath,
      Map<String, String> trainingParams) {
    // TODO Auto-generated method stub
    // train model with model trainer
  }

  @Override
  public void readFields(DataInput input) throws IOException {
    this.modelType = WritableUtils.readString(input);
    this.regularization = input.readDouble();
    this.costFunction = FunctionFactory
        .createDoubleDoubleFunction(WritableUtils.readString(input));

    // read input dimension size
    if (this.layerSizeList == null) {
      this.layerSizeList = new ArrayList<Integer>(2);
    }
    this.layerSizeList.set(0, input.readInt());

    // read weights
    if (this.weightMatrixList == null) {
      this.weightMatrixList = new ArrayList<DoubleMatrix>(1);
    }
    this.weightMatrixList
        .set(0, (DenseDoubleMatrix) MatrixWritable.read(input));

    // set default fields
    this.learningRate = 1.0;
    this.layerSizeList.set(1, 1);

    if (this.squashingFunctionList == null) {
      this.squashingFunctionList = new ArrayList<DoubleFunction>(1);
    }
    this.squashingFunctionList.set(0,
        FunctionFactory.createDoubleFunction("Identity"));

  }

  @Override
  public void write(DataOutput output) throws IOException {
    // TODO Auto-generated method stub
    WritableUtils.writeString(output, modelType);
    output.writeDouble(regularization);
    WritableUtils.writeString(output, costFunction.getFunctionName());
    // add size of input dimensions
    output.writeInt(this.layerSizeList.get(0));
    // add weights
    MatrixWritable matrixWritable = new MatrixWritable(
        this.weightMatrixList.get(0));
    matrixWritable.write(output);
  }

  @Override
  protected void setModelType() {
    this.modelType = "LocalLinearRegression";
  }

}
