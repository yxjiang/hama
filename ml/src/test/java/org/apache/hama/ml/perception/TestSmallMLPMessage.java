
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.junit.Test;


/**
 * Test the functionalities of SmallMLPMessage
 * 
 */
public class TestSmallMLPMessage {

  @Test
  public void testReadWrite() {
    int owner = 101;
    double[][] mat = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };

    double[][] mat2 = { { 10, 20 }, { 30, 40 }, { 50, 60 } };

    double[][][] mats = { mat, mat2 };

    DenseDoubleMatrix[] matrices = new DenseDoubleMatrix[] {
        new DenseDoubleMatrix(mat), new DenseDoubleMatrix(mat2) };

    SmallMLPMessage message = new SmallMLPMessage(owner, true, matrices);

    Configuration conf = new Configuration();
    String strPath = "testSmallMLPMessage";
    Path path = new Path(strPath);
    try {
      FileSystem fs = FileSystem.get(new URI(strPath), conf);
      FSDataOutputStream out = fs.create(path, true);
      message.write(out);
      out.close();

      FSDataInputStream in = fs.open(path);
      SmallMLPMessage outMessage = new SmallMLPMessage(0, false, null);
      outMessage.readFields(in);

      assertEquals(owner, outMessage.getOwner());
      DenseDoubleMatrix[] outMatrices = outMessage.getWeightsUpdatedMatrices();
      // check each matrix
      for (int i = 0; i < outMatrices.length; ++i) {
        double[][] outMat = outMessage.getWeightsUpdatedMatrices()[i]
            .getValues();
        for (int j = 0; j < outMat.length; ++j) {
          assertArrayEquals(mats[i][j], outMat[j], 0.0001);
        }
      }

      fs.delete(path, true);
    } catch (IOException e) {
      e.printStackTrace();
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }

  }
}
