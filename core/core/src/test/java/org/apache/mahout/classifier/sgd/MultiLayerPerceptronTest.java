/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

public final class MultiLayerPerceptronTest extends OnlineBaseTest {
  
  private Vector getTarget(double x1, double x2, double x3) {
    Vector target = new DenseVector(3);
    target.set(0, x1);
    target.set(1, x2);
    target.set(2, x3);
    return target;
  }
  
  @Test
  public void testForwardPass() throws IOException {
    
    int nbInputUnits = 3;
    int nbOutputUnits = 2;
    int hiddenUnits[] = {2};
    MultiLayerPerceptron mlp = new MultiLayerPerceptron(nbInputUnits,
        nbOutputUnits, hiddenUnits, false);
    double w0[][] = { {0, 0, 0, 0}, // to bias
        {20, -40, 0, 0}, {10, -20, 20, -40}};
    double w1[][] = { {10, 20, 40}, {-40, 30, 30}};
    Matrix weights[] = new DenseMatrix[2];
    weights[0] = new DenseMatrix(w0);
    weights[1] = new DenseMatrix(w1);
    mlp.initWeights(weights);
    // apply a forward pass
    Vector out = mlp.classifyFull(getTarget(1, 1, 0));
    assertTrue("MLP ForwardPass doesn't work!", out.get(0) > 0.999);
    assertTrue("MLP ForwardPass doesn't work!", out.get(1) < 0.001);
    
    out = mlp.classifyFull(getTarget(0, 1, 1));
    assertTrue("MLP ForwardPass doesn't work!", out.get(0) < 0.001);
    assertTrue("MLP ForwardPass doesn't work!", out.get(1) < 0.001);
  }
  
  private double getSumSquareWeigths(MultiLayerPerceptron mlp){
    Matrix [] weights = mlp.weights;
    double squareSum = 0;
    for(int i=0; i<weights.length ; i++){
      squareSum += weights[i].aggregate(Functions.PLUS, Functions.SQUARE);
    }
   return squareSum;
  }
  
  @Test
  public void testRegularization() throws IOException {
    Vector target = readStandardData();
    double regularization = 0;
    double oldSquareSum = Double.MAX_VALUE;
    for(int l = 0; l<99; l++){
      MultiLayerPerceptron mlp = getMlp().learningRate(0.01).momentum(0.1).regularization(regularization);
      for (int i = 0; i < 10; i++) {
        train(getInput(), target, mlp);
      }
      double squareSum = getSumSquareWeigths(mlp);
      assertTrue("error: weight decay", squareSum < oldSquareSum);
      //System.out.println("weightsSquareSum: " + squareSum);
      oldSquareSum = squareSum;
      regularization += 0.01;
    }
    
  }
  
  @Test
  public void testLearningCheck() throws IOException {
    Vector target = readStandardData();
  
    MultiLayerPerceptron mlp = getMlp().learningRate(0.001).momentum(0.0).regularization(0.0);
    RandomUtils.useTestSeed();
    Random gen = RandomUtils.getRandom();
    mlp.initWeightsRandomly(gen);
    Matrix input = getInput();
    for (int row = 0; row < 60; row++) {
      double tv = target.get(row);
      Vector t = new DenseVector(1);
      t.setQuick(0, tv);
      Vector i = input.viewRow(row);
      
      // train with only one pattern; error
      double oldCost = Double.MAX_VALUE;
      for (int j = 0; j < 10000; j++) {
        // print cost
        Vector out = mlp.trainOnline(i, t);
        double cost = mlp.getCost(out, t);
        //System.out.print("out " + out + "  target " + t);
        //System.out.println("  cost " + cost + "  oldCost " + oldCost);
        // high learning rate could produce an error
        assertTrue("error: MLP doesn't learn", oldCost >= cost);
        
        oldCost = cost;
      }
      mlp.initWeightsRandomly(gen);
    }
  }
  
  
  MultiLayerPerceptron getMlp(){
    int[] nbHiddenUnits = {5};
    int nbInputUnits = 8;
    int nbOutputUnits = 1;
    MultiLayerPerceptron mlp = new MultiLayerPerceptron(nbInputUnits,
        nbOutputUnits, nbHiddenUnits, false);
    RandomUtils.useTestSeed();
    Random gen = RandomUtils.getRandom();
    mlp.initWeightsRandomly(gen);
    return mlp;
  }
  
  final static double MAX_ABSOLUTE_TOLERANZ = 0.0001;
  final static double MAX_RELATIVE_TOLERANZ = 0.001;
  @Test 
  public void gradientCrossChecking() throws IOException{
    Vector allTarget = readStandardData();
    MultiLayerPerceptron mlp = getMlp();
    
    RandomUtils.useTestSeed();
    Random gen = RandomUtils.getRandom();
    Matrix allInput = getInput();
    // train on samples in random order (but only one pass)
    for (int row : permute(gen, 60)) {
      Vector input = allInput.viewRow(row);
      int targetValue = (int) allTarget.get(row);
      mlp.train(targetValue, input);
      Vector target = new DenseVector(1);
      target.setQuick(0, (double) targetValue);
      //if (gen.nextDouble() < 0.1){
        Matrix[] backGradient = 
            mlp.getDerivativeOfTheCostWithoutRegularization(input, target);
        Matrix[] numericalGradient =
            getGradientNumerically(mlp, input, target);
        //System.out.print("----------------");
        for(int i = 0; i<backGradient.length;i++){
          for(int l=0; l<backGradient[i].rowSize(); l++){
            for(int m=0; m<backGradient[i].columnSize(); m++){
              double bg = backGradient[i].get(l, m);
              double ng = numericalGradient[i].get(l, m);
              double diff = Math.abs( bg - ng);
              assertTrue("sign error in backprop", Math.signum(bg) == Math.signum(ng));
              assertTrue("error in backprop (at)", MAX_ABSOLUTE_TOLERANZ > diff);
              if(ng != 0d){
                assertTrue("error in backprop (rt)", MAX_RELATIVE_TOLERANZ > diff/ng);
              }
              //System.out.print("back: " + backGradient[i].get(l, m));
              //System.out.println("  numGra: " + numericalGradient[i].get(l, m));
            }
          } 
      }
    }
    mlp.close();
  }
  
  static final double epsilon = 0.0001;
  private Matrix[] getGradientNumerically(MultiLayerPerceptron mlp, Vector input, Vector target){
    Matrix [ ]gradientMatrices = mlp.getMatrixTopology(); 
    Vector output = mlp.classifyFull(input);
    //double cost  = getCost(output, target);
    Matrix []weights = mlp.weights;
    for(int i=0; i< weights.length; i++){
      for(int l=0; l<weights[i].rowSize(); l++){
        for(int m=0; m<weights[i].columnSize(); m++){
          // E(w_lm + epsilon)
          weights[i].set(l, m, weights[i].get(l, m) + epsilon);
          output = mlp.classifyFull(input);
          double cost_plus_w  = mlp.getCost(output, target);  
          weights[i].set(l, m, weights[i].get(l, m) - 2 * epsilon);
          output = mlp.classifyFull(input);
          double cost_minus_w  = mlp.getCost(output, target);  
          weights[i].set(l, m, weights[i].get(l, m) +  epsilon);    
          double gradient = (cost_plus_w - cost_minus_w)/(2 *epsilon);
          gradientMatrices[i].set(l, m, gradient);
          } 
        }
      }
    return gradientMatrices;
    }
  
  
  @Test
  public void testMultiLayerPerceptron() throws IOException {
    Vector target = readStandardData();
    MultiLayerPerceptron mlp = getMlp();
    for (int i = 0; i < 1; i++) {
      train(getInput(), target, mlp);
    }
    test(getInput(), target, mlp, 0.3, 1);
  }
  
}
