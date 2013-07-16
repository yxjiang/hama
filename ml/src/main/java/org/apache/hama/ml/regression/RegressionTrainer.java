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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.ml.ann.SmallLayeredNeuralNetwork;
import org.apache.hama.ml.ann.SmallLayeredNeuralNetworkMessage;
import org.apache.hama.ml.ann.SmallLayeredNeuralNetworkTrainer;
import org.apache.hama.ml.math.DoubleDoubleFunction;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.FunctionFactory;
import org.apache.hama.ml.writable.VectorWritable;

/**
 * RegressionTrainer is used for both LinearRegression and LogisticRegression.
 * 
 */
public final class RegressionTrainer extends SmallLayeredNeuralNetworkTrainer {

  private int maxIteration;
  private int curIteration;
  private boolean terminateTraining;

  @Override
  public void extraSetup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer) {
    maxIteration = conf.getInt("training.iteration", 1000);
    String existingModelPath = conf.get("existingModelPath");
    if (existingModelPath != null) {
      this.inMemoryModel = new LinearRegression(existingModelPath);
    } else {
      double learningRate = Double.parseDouble(conf.get("learningRate", ""
          + SmallLayeredNeuralNetwork.DEFAULT_LEARNING_RATE));
      double regularizationWeight = Double.parseDouble(conf.get(
          "regularizationWeight", ""
              + SmallLayeredNeuralNetwork.DEFAULT_REGULARIZATION_WEIGHT));
      String squashingFunction = conf.get("squashingFunction");
      DoubleDoubleFunction costFunction = FunctionFactory
          .createDoubleDoubleFunction(conf.get("costFunction"));
      this.inMemoryModel = new LinearRegression(Integer.parseInt(conf
          .get("dimension")));
      this.inMemoryModel.setLearningRate(learningRate);
      this.inMemoryModel.setRegularizationWeight(regularizationWeight);
      this.inMemoryModel.setCostFunction(costFunction);
      this.inMemoryModel.setSquashingFunction(FunctionFactory
          .createDoubleFunction(squashingFunction));
    }
  }

  @Override
  public void calculateUpdates(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer) {
    // TODO Auto-generated method stub

  }

  @Override
  public void mergeUpdates(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException {
    DoubleMatrix[] matrices = this.getZeroWeightMatrices();
    int numPartitions = peer.getNumCurrentMessages();
    while (peer.getNumCurrentMessages() > 0) {
      SmallLayeredNeuralNetworkMessage message = peer.getCurrentMessage();
      DoubleMatrix[] updates = message.getWeightUpdates();
      LinearRegression.matricesAdd(matrices, updates);
    }

    for (int i = 0; i < matrices.length; ++i) {
      matrices[i] = matrices[i].divide(numPartitions);
    }

    this.inMemoryModel.updateWeightMatrices(matrices);

    // send updated matrix to all grooms
    for (int i = 0; i < numPartitions; ++i) {
      SmallLayeredNeuralNetworkMessage message = new SmallLayeredNeuralNetworkMessage(
          peer.getPeerIndex(), this.terminateTraining, this.inMemoryModel.getWeightMatrices(), null);
      peer.send(peer.getPeerName(i), message);
    }
  }

  @Override
  public boolean checkTerminated() {
    // TODO Auto-generated method stub
    if (curIteration >= maxIteration) {
      return true;
    }
    return false;
  }

  @Override
  protected void extraCleanup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException {
    // TODO Auto-generated method stub

  }

}
