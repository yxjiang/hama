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
import java.util.BitSet;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.writable.VectorWritable;

import com.google.common.base.Preconditions;

/**
 * SmallLayeredNeuralNetworkTrainer defines the common behaviors for
 * {@link SmallLayeredNeuralNetwork}.
 * 
 */
public abstract class SmallLayeredNeuralNetworkTrainer
    extends
    BSP<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> {

  private static final Log LOG = LogFactory
      .getLog(SmallLayeredNeuralNetworkTrainer.class);

  protected Configuration conf;
  /* used by master only, check whether all slaves finishes reading */
  protected BitSet statusSet;
  /* Once reader reaches the EOF, the training procedure would be terminated */
  private boolean terminateTraining = false;
  /* the in-memory model that maintain the up-to-date parameters globally */
  protected SmallLayeredNeuralNetwork inMemoryModel;
  protected int batchSize;
  /* indicate where to store the trained model */
  protected String modelPath;

  @Override
  /**
   * Obtain the training related parameters.
   */
  final public void setup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException, SyncException, InterruptedException {
    // read general parameters
    conf = peer.getConfiguration();
    this.batchSize = conf.getInt("training.batch.size", 100);
    this.statusSet = new BitSet(peer.getConfiguration().getInt("tasks", 1));
    this.modelPath = conf.get("modelPath");
    Preconditions
        .checkArgument(modelPath != null,
            "Please specify path to store the model with 'modelPath' property name.");
    // read model specific parameters
    this.extraSetup(peer);
  }

  /**
   * Extra setup for specific trainer. The in-memory model should be initialized
   * in this method.
   * 
   * @param peer
   */
  public abstract void extraSetup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer);

  /**
   * {@inheritDoc}
   */
  @Override
  public void bsp(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException, SyncException, InterruptedException {
    LOG.info("Start training...");
    while (terminateTraining) {
      calculateUpdates(peer); // each peer calculate updates locally
      peer.sync();

      if (peer.getPeerIndex() == 0) {
        mergeUpdates(peer);
      }
      peer.sync();

      if (checkTerminated()) {
        break;
      }
    }
    LOG.info(String.format("Task %d finished.", peer.getPeerIndex()));
  }

  /**
   * Calculate the weight updates locally.
   */
  public abstract void calculateUpdates(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException;

  /**
   * Master merges all the updates. Then sends the updated matrices as well as
   * associated information to all grooms.
   */
  public abstract void mergeUpdates(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException;

  /**
   * Identify whether to terminate the training.
   * 
   * @return
   */
  public abstract boolean checkTerminated();

  @Override
  /**
   * Write the model to specified location, indicated by 'training.output.path'
   */
  public void cleanup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException {
    this.extraCleanup(peer);
    // write model to modelPath
    LOG.info(String.format("Write model back to %s\n", this.modelPath));
    this.inMemoryModel.writeModelToFile(this.modelPath);
  }

  /**
   * Handle cleanup for sub-classes. Write the trained model back.
   * 
   * @param peer
   * @throws IOException
   * @throws SyncException
   * @throws InterruptedException
   */
  protected abstract void extraCleanup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException;

  /**
   * Initialize the weight matrices.
   */
  protected DenseDoubleMatrix[] getZeroWeightMatrices() {
    List<Integer> layerSizeArray = this.inMemoryModel.getLayerSizeList();
    DenseDoubleMatrix[] weightUpdateCache = new DenseDoubleMatrix[layerSizeArray
        .size() - 1];
    // initialize weight matrix each layer
    for (int i = 0; i < weightUpdateCache.length; ++i) {
      weightUpdateCache[i] = new DenseDoubleMatrix(layerSizeArray.get(i) + 1,
          layerSizeArray.get(i + 1));
    }
    return weightUpdateCache;
  }

}
