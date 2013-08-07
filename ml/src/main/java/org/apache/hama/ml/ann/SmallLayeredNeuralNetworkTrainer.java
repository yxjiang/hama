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

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.math.DoubleVector;
import org.apache.hama.ml.writable.VectorWritable;
import org.mortbay.log.Log;

/**
 * The trainer that train the {@link SmallLayeredNeuralNetwork} based on BSP
 * framework.
 * 
 */
public final class SmallLayeredNeuralNetworkTrainer
    extends
    BSP<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> {

  private SmallLayeredNeuralNetwork inMemoryModel;
  private Configuration conf;
  /* Default batch size */
  private int batchSize;

  /* check the interval between intervals */
  private double prevAvgTrainingError;
  private double curAvgTrainingError;
  private long convergenceCheckInterval;
  private long iterations;
  private long maxIterations;

  private String modelPath;

  @Override
  /**
   * If the model path is specified, load the existing from storage location.
   */
  public void setup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer) {
    Log.info("Begin to train");
    this.conf = peer.getConfiguration();
    this.iterations = 0;
    this.modelPath = conf.get("modelPath");
    this.maxIterations = conf.getLong("training.max.iterations", 100000);
    this.convergenceCheckInterval = conf.getLong("convergence.check.interval",
        1000);
    this.modelPath = conf.get("modelPath");
    this.inMemoryModel = new SmallLayeredNeuralNetwork(modelPath);
    this.prevAvgTrainingError = Integer.MAX_VALUE;
    this.batchSize = conf.getInt("training.batch.size", 100);
  }

  @Override
  /**
   * Write the trained model back to stored location.
   */
  public void cleanup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer) {
    // write model to modelPath
    if (peer.getPeerIndex() == 0) {
      try {
        Log.info(String.format("End of training, number of iterations: %d.\n",
            this.iterations));
        System.out.printf("Write model back to %s\n", inMemoryModel.getModelPath());
        this.inMemoryModel.writeModelToFile();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  @Override
  public void bsp(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException, SyncException, InterruptedException {
    while (this.iterations++ < maxIterations) {
      
      // each groom calculate the matrices updates according to local data
      calculateUpdates(peer);
      peer.sync();

      // master merge the updates model
      if (peer.getPeerIndex() == 0) {
        mergeUpdates(peer);
      }
      peer.sync();
    }
  }

  /**
   * Calculate the matrices updates according to local partition of data.
   * 
   * @param peer
   * @throws IOException
   */
  private void calculateUpdates(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException {
    // receive update information from master
    if (peer.getNumCurrentMessages() != 0) {
      SmallLayeredNeuralNetworkMessage message = peer.getCurrentMessage();
      DoubleMatrix[] curMatrix = message.getCurMatrices();
      this.inMemoryModel.setWeightMatrices(curMatrix);
      boolean isConverge = message.isConverge();
      if (isConverge) {
        return;
      }
    }

    // continue to train
    int recordsRead = 0;
    double avgTrainingError = 0.0;
    LongWritable key = new LongWritable();
    VectorWritable value = new VectorWritable();
    DoubleMatrix[] weightUpdates = new DoubleMatrix[this.inMemoryModel.weightMatrixList.size()];
    for (int i = 0; i < weightUpdates.length; ++i) {
      int row = this.inMemoryModel.weightMatrixList.get(i).getRowCount();
      int col = this.inMemoryModel.weightMatrixList.get(i).getColumnCount();
      weightUpdates[i] = new DenseDoubleMatrix(row, col);
    }
    
    while (recordsRead++ < batchSize) {
      if (peer.readNext(key, value) == false) {
        peer.reopenInput();
        peer.readNext(key, value);
      }
      DoubleVector trainingInstance = value.getVector();
      SmallLayeredNeuralNetwork.matricesAdd(weightUpdates, this.inMemoryModel.trainByInstance(trainingInstance));
      avgTrainingError += this.inMemoryModel.trainingError / batchSize;
    }
    
    // calculate the average of updates
    for (int i = 0; i < weightUpdates.length; ++i) {
      weightUpdates[i] = weightUpdates[i].divide(batchSize);
    }
    
    DoubleMatrix[] prevWeightUpdates = this.inMemoryModel.getPrevMatricesUpdates();
    SmallLayeredNeuralNetworkMessage outMessage = new SmallLayeredNeuralNetworkMessage(
        avgTrainingError, false, weightUpdates, prevWeightUpdates);
    peer.send(peer.getPeerName(0), outMessage);
  }

  /**
   * Merge the updates according to the updates of the grooms.
   * 
   * @param peer
   * @throws IOException
   */
  private void mergeUpdates(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException {
    int numMessages = peer.getNumCurrentMessages();

    if (numMessages == 0) { // converges
      this.inMemoryModel.setConverge(true);
      return;
    }

    double avgTrainingError = 0;
    DoubleMatrix[] matricesUpdates = null;
    DoubleMatrix[] prevMatricesUpdates = null;

    while (peer.getNumCurrentMessages() > 0) {
      SmallLayeredNeuralNetworkMessage message = peer.getCurrentMessage();
      if (matricesUpdates == null) {
        matricesUpdates = message.getCurMatrices();
        prevMatricesUpdates = message.getPrevMatrices();
      } else {
        SmallLayeredNeuralNetwork.matricesAdd(matricesUpdates, message.getCurMatrices());
        SmallLayeredNeuralNetwork.matricesAdd(prevMatricesUpdates, message.getPrevMatrices());
      }
      avgTrainingError += message.getTrainingError() / numMessages;
    }

    if (numMessages != 1) {
      for (int i = 0; i < matricesUpdates.length; ++i) {
        matricesUpdates[i] = matricesUpdates[i].divide(numMessages);
        prevMatricesUpdates[i] = prevMatricesUpdates[i].divide(numMessages);
      }
    }
    this.inMemoryModel.updateWeightMatrices(matricesUpdates);
    this.inMemoryModel.setPrevWeightMatrices(prevMatricesUpdates);

    // check convergence
    if (iterations % convergenceCheckInterval == 0) {
      System.out.printf("Prev avg error %f, cur avg error %f\n", prevAvgTrainingError, curAvgTrainingError);
      if (prevAvgTrainingError < curAvgTrainingError) {
        // error cannot decrease any more
        this.inMemoryModel.setConverge(true);
      } 
      // update
      prevAvgTrainingError = curAvgTrainingError;
      curAvgTrainingError = 0;
    }
    curAvgTrainingError += avgTrainingError / convergenceCheckInterval;

    // broadcast updated weight matrices
    for (String peerName : peer.getAllPeerNames()) {
      SmallLayeredNeuralNetworkMessage msg = new SmallLayeredNeuralNetworkMessage(
          0, this.inMemoryModel.isConverge(),
          this.inMemoryModel.getWeightMatrices(),
          this.inMemoryModel.getPrevMatricesUpdates());
      peer.send(peerName, msg);
    }
  }

}