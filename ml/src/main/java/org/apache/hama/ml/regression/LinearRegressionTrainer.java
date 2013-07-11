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
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.ml.ann.SmallLayeredNeuralNetworkTrainer;
import org.apache.hama.ml.perception.MLPMessage;
import org.apache.hama.ml.writable.VectorWritable;

/**
 * 
 *
 */
public final class LinearRegressionTrainer extends SmallLayeredNeuralNetworkTrainer {
  
  // the in-memory model maintained in each groom task
  private LinearRegression inMemoryLinearRegressionModel;

  @Override
  public void bsp(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, MLPMessage> peer)
      throws IOException, SyncException, InterruptedException {
    // TODO Auto-generated method stub
    
  }

}
