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

package org.apache.hama.ml.semiclustering;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hama.graph.Edge;
import org.apache.hama.graph.Vertex;
import org.apache.hama.graph.VertexInputReader;

import java.util.ArrayList;
import java.util.List;

/**
 * SemiClusterTextReader defines the way in which data is to be read from the
 * input file and store as a vertex with VertexId and Edges
 * 
 */
public class SemiClusterTextReader extends
    VertexInputReader<LongWritable, Text, Text, DoubleWritable, Text> {

  String lastVertexId = null;
  List<String> adjacents = new ArrayList<String>();

  @Override
  public boolean parseVertex(LongWritable key, Text value,
      Vertex<Text, DoubleWritable, Text> vertex) {

    String line = value.toString();
    String[] lineSplit = line.split("\t");
    if (!line.startsWith("#")) {
      if (lastVertexId == null) {
        lastVertexId = lineSplit[0];
      }
      if (lastVertexId.equals(lineSplit[0])) {
        adjacents.add(lineSplit[1]);
      } else {
        vertex.setVertexID(new Text(lastVertexId));
        for (String adjacent : adjacents) {
          String[] ValueSplit = adjacent.split("-");
          vertex.addEdge(new Edge<Text, DoubleWritable>(
              new Text(ValueSplit[0]), new DoubleWritable(Double
                  .parseDouble(ValueSplit[1]))));
        }
        adjacents.clear();
        lastVertexId = lineSplit[0];
        adjacents.add(lineSplit[1]);
        return true;
      }
    }
    return false;
  }

}
