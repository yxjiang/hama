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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.writable.MatrixWritable;

/**
 * NeuralNetworkMessage transmits the messages between peers during the training
 * of neural networks.
 * 
 */
public class SmallLayeredNeuralNetworkMessage implements Writable {

  protected int ownerIdx;
  protected boolean terminated;
  protected DoubleMatrix[] curMatrices;
  protected DoubleMatrix[] prevMatrices;
  
  public SmallLayeredNeuralNetworkMessage(int ownerIdx, boolean terminated, DoubleMatrix[] weightMatrices, DoubleMatrix[] prevMatrices) {
    this.ownerIdx = ownerIdx;
    this.terminated = terminated;
    this.curMatrices = weightMatrices;
    this.prevMatrices = prevMatrices;
  }

  @Override
  public void readFields(DataInput input) throws IOException {
    terminated = input.readBoolean();
    int numMatrices = input.readInt();
    curMatrices = new DenseDoubleMatrix[numMatrices];
    prevMatrices = new DenseDoubleMatrix[numMatrices];
    // read matrice updates
    for (int i = 0; i < curMatrices.length; ++i) {
      curMatrices[i] = (DenseDoubleMatrix) MatrixWritable.read(input);
    }
    boolean hasPrevMatrices = input.readBoolean();
    if (hasPrevMatrices) {
      // read previous matrices updates
      for (int i = 0; i < prevMatrices.length; ++i) {
        prevMatrices[i] = (DenseDoubleMatrix) MatrixWritable.read(input);
      }
    }
  }

  @Override
  public void write(DataOutput output) throws IOException {
    output.writeBoolean(terminated);
    output.writeInt(curMatrices.length);
    for (int i = 0; i < curMatrices.length; ++i) {
      MatrixWritable.write(curMatrices[i], output);
    }
    if (prevMatrices != null) {
      output.writeBoolean(true);
      for (int i = 0; i < prevMatrices.length; ++i) {
        MatrixWritable.write(prevMatrices[i], output);
      }
    }
    else {
      // add mark if preMatrices is null
      output.writeBoolean(false);
    }
  }

  public void setTerminated(boolean terminated) {
    this.terminated = terminated;
  }

  public boolean isTerminated() {
    return terminated;
  }

  public DoubleMatrix[] getCurMatrices() {
    return curMatrices;
  }

  public void setMatrices(DoubleMatrix[] curMatrices) {
    this.curMatrices = curMatrices;
  }

  public DoubleMatrix[] getPrevMatrices() {
    return prevMatrices;
  }

  public void setPrevMatrices(DoubleMatrix[] prevMatrices) {
    this.prevMatrices = prevMatrices;
  }
  
}
