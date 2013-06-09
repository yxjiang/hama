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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Random;


/**
 * Multilayer Perceptron MLP
 * 
 * Implements MLP with arbitrary number of hidden layers. Only neurons/units
 * between layers are connected. In future: IO-Short cuts for faster learning
 * (of the linear model)??
 * 
 * Weights as matrices. Each layer (except the output layer) has a additional
 * bias unit +1 The forward pass can be done with: a^(l+1) = g(W^(l) * a^(l)) =
 * g(z^(l)) with a^(l): Vector of the activations of the units in layer l resp.
 * l+1; W: weight matrix
 * 
 */
public class MultiLayerPerceptron extends AbstractVectorClassifier implements
    OnlineLearner, Writable {
  
  public static final int WRITABLE_VERSION = 1;
  
  // the learning rate of the algorithm
  protected double learningRate = .1;
  
  // the regularization term, a positive number that controls the size of the
  // weight vector
  private double regularization = 0.01;
  
  //
  private double momentum = 0.8;
  
  // the number of hidden layer
  protected int nbLayer;
  
  // weight matrices
  protected Matrix[] weights;
  
  // Matrices for storing delta w
  // needed for momentum term
  private Matrix[] oldWeightChange;
  
  // container for the activations
  protected Vector[] units;
  
  // number of units in each layer: input layer is layer 0
  protected int[] nbUnits;
  // Convenience
  private int nbOutputs;
  
  private boolean mutuallyExclusiveClasses;
  
  private boolean hasNaturalPairing = true;
  
  // the activity functions res. gradients of each layer
  // activityFunction.lenght = nbLayers, for layer 0 this will not be used
  // so the same indices can be uses as for the layers
  private Squashing squashingFunctions[];
  
  // private DoubleFunction activityFunctions[];
  // private DoubleFunction activityGradients[];
  // there is a natural paring between the activity function of the output
  // units and the error function
  // DoubleFunction costFunction;
  
  public enum Squashing {
    //
    LINEAR {
      @Override
      Vector apply(Vector v) {
        return v;
      }
      
      @Override
      Vector applyGradient(Vector v) {
        throw new UnsupportedOperationException();
      }
    },
    SIGMOID // standard output shashing for classification of n idependent
    // targets
    {
      @Override
      Vector apply(Vector v) {
        return v.assign(Functions.SIGMOID);
      }
      
      @Override
      Vector applyGradient(Vector v) {
        return v.assign(Functions.SIGMOIDGRADIENT);
      }
    },
    
    TANH // standard squashing of the hidden units
    { // TODO modify tanh, see Yann LeCun et. al.: "Efficient BackProp"
      @Override
      Vector apply(Vector v) {
        return v.assign(Functions.TANH);
      }
      
      @Override
      Vector applyGradient(Vector v) {
        return v.assign(Functions.TANHGRADIENT);
      }
    },
    
    SOFTMAX // standard output squashing for classification of n mutually
    // exclusive classes
    { //
      @Override
      Vector apply(Vector v) {
        double partitionFunction = v.aggregate(Functions.PLUS, Functions.EXP);
        v.assign(Functions.EXP);
        return v.assign(Functions.DIV, partitionFunction);
      }
      
      // gradient should never be used:
      @Override
      Vector applyGradient(Vector v) {
        throw new UnsupportedOperationException();
      }
    };
    /**
     * Apply the squashing / activation function
     * 
     * @param v
     * @return
     */
    abstract Vector apply(Vector v);
    
    /**
     * Apply the derivation of the squashing / activation function
     * 
     * @param a
     *          : already the activations of the units
     * @return
     */
    abstract Vector applyGradient(Vector a);
  }
  
  protected CostFunction costFunction;
  
  public enum CostFunction {
    
    // TODO SQUARD_ERROR for regression in combination with linear squashing
    CROSS_ENTROPY_INDEPENDENT_OUTPUTS {
      @Override
      /**
       * E = sum_o ( t_o ln(y_o) * (1-t_o ln (1-y_o))
       * with sum_o: sum over all output units  
       */
      double getCost(Vector output, Vector target) // throws
      // InvalidArgumentException
      {
        double e = 0;
        // TODO make this numerically more stable
        // log(0) = - infinity => should be avoided
        for (int i = 0; i < output.size(); i++) {
          if (target.getQuick(i) == output.getQuick(i)) continue;
          if (target.get(i) == 1) {
            e -= Math.log(output.get(i));
          } else if (target.get(i) == 0) {
            e -= Math.log(1 - output.get(i));
          } else { // TODO
            // throw new
            // InvalidArgumentException("target value != 0 or 1");
          }
        }
        ;
        return e;
      }
    },
    CROSS_ENTROPY_MUTUALLY_EXCUSIVE_OUTPUTS {
      @Override
      double getCost(Vector output, Vector target) // throws
      // InvalidArgumentException
      {
        double e = 0;
        // TODO make this numerically more stable
        // log(0) = - infinity => should be avoided
        for (int i = 0; i < output.size(); i++) {
          if (target.get(i) == 1) {
            e -= Math.log(output.get(i));
          }
        }
        return e;
      }
    };
    /**
     * get the cost for the error function without regularization cost
     * 
     * @param output
     * @param target
     * @return
     * 
     */
    abstract double getCost(Vector output, Vector target);// throws
    // InvalidArgumentException;
  }
  
  /**
   * Construction of nbLayer -1 matrices which can hold weights, weightChanges
   * etc.
   * 
   * @return
   */
  protected Matrix[] getMatrixTopology() {
    Matrix[] w = new DenseMatrix[nbLayer - 1];
    for (int i = 0; i < nbLayer - 1; i++) {
      //
      w[i] = new DenseMatrix(nbUnits[i + 1], nbUnits[i]);
    }
    return w;
  }
  
  /**
   * Construction of nbLayer vectors which can hold activations, deltas etc.
   * 
   * @return
   */
  private Vector[] getVectorTopology() {
    // construction of the weight matrices
    Vector[] v = new DenseVector[nbLayer];
    for (int i = 0; i < nbLayer; i++) {
      v[i] = new DenseVector(nbUnits[i]);
    }
    return v;
  }
  
  private void initResidue() {
    // set topology info:
    nbUnits = new int[nbLayer];
    for (int i = 0; i < nbLayer - 1; i++) {
      nbUnits[i] = weights[i].numCols();
    }
    nbUnits[nbLayer - 1] = weights[nbLayer - 2].numRows();
    nbOutputs = nbUnits[nbLayer - 1];
    
    // construction of the units
    units = getVectorTopology();
    oldWeightChange = getMatrixTopology();
  }
  
  private void init(int nbInputUnits, int nbOutputUnits, int[] nbHiddenUnits) {
    
    if (nbHiddenUnits != null) {
      nbLayer = nbHiddenUnits.length + 2;
    } else {
      nbLayer = 2;
    }
    
    nbUnits = new int[nbLayer];
    
    // + 1 for the bias unit
    nbUnits[0] = nbInputUnits + 1;
    // no bias at output
    nbUnits[nbLayer - 1] = nbOutputUnits;
    nbOutputs = nbOutputUnits;
    for (int i = 1; i < nbLayer - 1; i++) {
      nbUnits[i] = nbHiddenUnits[i - 1] + 1;
    }
    
    // construction of all weights
    weights = getMatrixTopology();
    
    // construction of the units
    units = getVectorTopology();
    
    oldWeightChange = getMatrixTopology();
  }
  
  /**
   * Constructor for MLP weights are not initialize before learning they must be
   * initialized by initWeightsRandomly()
   * 
   * @param nbInputUnits
   * @param nbOutputUnits
   * @param nbHiddenUnits
   *          Array which holds the number of hidden units without counting bias
   *          units, e.g. nbHiddenUnits[2]={6,4}; 6 hidden units in layer 1 and
   *          4 units in layer 2
   * @param mutuallyExclusiveClasses
   */
  public MultiLayerPerceptron(int nbInputUnits, int nbOutputUnits,
      int[] nbHiddenUnits, boolean mutuallyExclusiveClasses) {
    this.mutuallyExclusiveClasses = mutuallyExclusiveClasses;
    init(nbInputUnits, nbOutputUnits, nbHiddenUnits);
    setDefaultClassificationActivities(mutuallyExclusiveClasses);
  }
  
  /**
   * 
   * @return
   */
  private void setDefaultClassificationActivities(
      boolean mutuallyExclusiveClasses) {
    
    hasNaturalPairing = true;
    squashingFunctions = new Squashing[nbLayer];
    
    // hidden layer have tanh activations (TODO! modify tanh according to paper
    // accordingly)
    // should't be sigmoid; see Yann LeCun et. al.: "Efficient BackProp"
    for (int i = 1; i < nbLayer - 1; i++) {
      squashingFunctions[i] = Squashing.TANH;
    }
    if (mutuallyExclusiveClasses) {
      squashingFunctions[nbLayer - 1] = Squashing.SOFTMAX;
      costFunction = CostFunction.CROSS_ENTROPY_MUTUALLY_EXCUSIVE_OUTPUTS;
    } else {
      // output layer is sigmoid (logistic function)
      squashingFunctions[nbLayer - 1] = Squashing.SIGMOID;
      costFunction = CostFunction.CROSS_ENTROPY_INDEPENDENT_OUTPUTS;
    }
    // return activities;
  }
  
  private Vector forwardPass(Matrix w, Vector i, Squashing squashing, int layer) {
    Vector o = w.times(i);
    squashing.apply(o);
    // unit 0 is the bias unit; set bias=1 except for output 
    if (layer != nbLayer - 2) {
      o.setQuick(0, 1.0);
    }
    return o;
  }
  
  /**
   * 
   * @param v
   * @return
   */
  private Vector addBias(Vector v) {
    int s = v.size() + 1;
    Vector out = new DenseVector(s);
    out.setQuick(0, 1.0);
    for (int i = 1; i < s; i++) {
      out.setQuick(i, v.get(i - 1));
    }
    return out;
  }
  
  /**
   * Just for prediction not for learning!
   * 
   * @param input
   * @return
   */
  private Vector forwardPropagation(Vector input) {
    Vector out = addBias(input);
    // nbWeightMatrices = nbLayer - 1
    for (int i = 0; i < nbLayer - 1; i++) {
      out = forwardPass(weights[i], out, squashingFunctions[i + 1], i);
    }
    return out;
  }
  
  /**
   * Just for prediction not for learning!
   * 
   * @param input
   * @return
   */
  private Vector forwardPropagationNoLink(Vector input) {
    Vector out = addBias(input);
    // nbWeightMatrices = nbLayer - 1
    for (int i = 0; i < nbLayer - 2; i++) {
      out = forwardPass(weights[i], out, squashingFunctions[i + 1], i);
    }
    out = forwardPass(weights[nbLayer - 2], out, Squashing.LINEAR, nbLayer - 2);
    return out;
  }
  
  /**
   * Forward pass to set all activations
   */
  private Vector setUnitsWithForwardPropagation(Vector input) {
    Vector out = addBias(input);
    units[0] = out;
    for (int i = 0; i < nbLayer - 1; i++) {
      out = forwardPass(weights[i], out, squashingFunctions[i + 1], i);
      units[i + 1] = out;
    }
    return out;
  }
  
  /**
   * @param delta
   *          vector of delta of layer l+1
   * @param wt
   *          transpose of weight matrix of layer l
   * @param a
   *          activities of layer l
   * @return vector delta of the layer l
   */
  private Vector getDeltaWithBackwardPass(Matrix wt, Vector delta, int l) {
    Vector d = wt.times(delta);
    // d = g'(z_j) * w^t * delta_k
    Vector DerivationOfActivations = units[l].clone();
    squashingFunctions[l].applyGradient(DerivationOfActivations);
    d.assign(DerivationOfActivations, Functions.MULT);
    return d;
  }
  
  /**
   * Can be used only if the natural pairing between cost function and output
   * unit activation function is valid. Then there is a simple formula for the
   * deltas (errors) d for the output neurons: d = y - t. with y: output
   * activity and t: target value For the natural pairing (conjugate link
   * functions) see e.g. C. Bishop: "Pattern Recognition and Machine Learning",
   * chapter 5.2 or more comprehensive in C. Bishop:
   * "Neural Networks for Pattern Recognition", chapter 6
   * 
   * @param targets
   * @return
   */
  private Vector getOutputDeltasForNaturalPairing(Vector targets) {
    Vector d = units[nbLayer - 1].clone();
    d.assign(targets, Functions.MINUS);
    return d;
  }
  
  private Vector getOutputDeltas(Vector targets) {
    if (hasNaturalPairing) {
      return getOutputDeltasForNaturalPairing(targets);
    } else {
      // could be implemented for special cases
      throw new UnsupportedOperationException();
    }
  }
  
  /*
   * Get the derivate of the cost for the given data pattern without
   * regularization
   * 
   * @param input
   *          the input pattern
   * @param target
   *          the should-be output pattern
   * @return the derivative of the total cost dE/dw for all w (weights)
   */
  protected Matrix[] getDerivativeOfTheCostWithoutRegularization(Vector input,
      Vector target) {
    
    // Optimization-TODO: check if initialization is expensive
    Vector delta[] = getVectorTopology();
    
    // 1) forward pass to find the activations of all hidden and output
    // units
    setUnitsWithForwardPropagation(input);
    
    // 2) Evaluate the deltas for the output units
    delta[nbLayer - 1] = getOutputDeltas(target);
    
    // 3) backpropagate the deltas to obtain the deltas for the hidden units
    // for the input units there are no deltas => i > 0
    for (int i = nbLayer - 2; i > 0; i--) {
      Matrix wt = weights[i].transpose();
      delta[i] = getDeltaWithBackwardPass(wt, delta[i + 1], i);
      // no delta for bias
      // delta[i].setQuick(0, 0.0);
    }
    
    // 4) set dE/dw_ij^(l) = delta_i * activations_j
    Matrix costDervative[] = getMatrixTopology();
    for (int i = 0; i < nbLayer - 1; i++) {
      // the cross product of mahout is the outer product
      costDervative[i] = delta[i + 1].cross(units[i]);
    }
    return costDervative;
  }
  
  private void setWeightChangeToBiasZero(Matrix m) {
    // set weight change to bias units zero
    // except for output nodes
    for (int j = 0; j < m.numCols(); j++) {
      m.setQuick(0, j, 0.0);
    }
  }
  
  /**
   * 
   * @param input
   * @param target
   * @return output vector
   */
  public Vector trainOnline(Vector input, Vector target) {
    Matrix[] weightChange = getDerivativeOfTheCostWithoutRegularization(input,
        target);
    for (int i = 0; i < nbLayer - 1; i++) {
      weightChange[i].assign(weights[i], new DoubleDoubleFunction() {
        @Override
        public double apply(double a, double b) {
          return a + regularization * b;
        }
      });
      if(i!= nbLayer-2){ // not to last output unit
        setWeightChangeToBiasZero(weightChange[i]);
      }
      Matrix wC = weightChange[i].times(-1.0 * learningRate);
      // momentum term
      Matrix m = oldWeightChange[i].clone();
      m = m.times(momentum);
      wC = wC.plus(m);
      // change weights according to weightChangeMatrix
      weights[i].assign(wC, Functions.PLUS);
      oldWeightChange[i] = wC;
    }
    /*
     * System.out.print("input" + input.toString()); System.out.print("target" +
     * target.toString()); System.out.print("output" + units[nbLayer -
     * 1].toString()); System.out.println("Cost: " +
     * costFunction.getCost(units[nbLayer - 1], target));
     */
    
    // return output
    return units[nbLayer - 1];
  }
  
  /**
   * Random initialization of the weight matrices according to 4.6 of Yann LeCun
   * et. al.: "Efficient BackProp"
   */
  public void initWeightsRandomly(Random gen) {
    
    for (int l = 0; l < nbLayer - 1; l++) {
      int fanIn = nbUnits[l];
      int fanOut = nbUnits[l + 1];
      for (int i = 0; i < fanOut; i++) {
        for (int j = 0; j < fanIn; j++) {
          double w = 0;
          // weights to bias units have to be zero! not for outputs
          // (no output bias)!
          if (i != 0 || l == nbLayer - 2) {
            // the standard derivation s of a uniform distribution
            // [-a,a] is 1/sqrt(3) * a
            // s_should = 1 / sqrt(fanIn) = 1 / sqrt(3) * a => a =
            // sqrt(3)/sqrt(fanIn)
            w = 1.73d / Math.sqrt(fanIn);
            w = (2.0 * gen.nextDouble() - 1.0) * w;
          }
          weights[l].setQuick(i, j, w);
        }
      }
    }
  }
  
  private void initWeights(Matrix w, int layer) {
    if (w.numCols() != weights[layer].numCols()
        || w.numRows() != weights[layer].numRows()) {
      throw new IllegalArgumentException();
    }
    weights[layer] = w.clone();
  }
  
  public void initWeights(Matrix[] w) {
    if (w.length != weights.length) {
      throw new IllegalArgumentException();
    }
    for (int i = 0; i < w.length; i++) {
      initWeights(w[i], i);
    }
  }
  
  /**
   * Chainable configuration option.
   * 
   * @param learningRate
   *          New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public MultiLayerPerceptron learningRate(double learningRate) {
    this.learningRate = learningRate;
    return this;
  }
  
  /**
   * Chainable configuration option.
   * 
   * @param regularization
   *          A positive value that controls the weight vector size.
   * @return This, so other configurations can be chained.
   */
  public MultiLayerPerceptron regularization(double regularization) {
    this.regularization = regularization;
    return this;
  }
  
  /**
   * Chainable configuration option.
   * 
   * @param momentum
   *          A positive value that controls the momentum.
   * @return This, so other configurations can be chained.
   */
  public MultiLayerPerceptron momentum(double momentum) {
    this.momentum = momentum;
    return this;
  }
  
  public MultiLayerPerceptron copy() {
    close();
    int[] nbHiddenUnits = new int[nbUnits.length - 2];
    for (int i = 0; i < nbHiddenUnits.length; i++) {
      nbHiddenUnits[i] = nbUnits[i + 1] - 1;
    }
    MultiLayerPerceptron mlp = new MultiLayerPerceptron(nbUnits[0] - 1,
        nbOutputs, nbHiddenUnits, mutuallyExclusiveClasses);
    mlp.copyFrom(this);
    return mlp;
  }
  
  @Override
  public int numCategories() {
    // TODO : check what numCategories means and fix this
    if (nbOutputs == 1) return 2;
    return nbOutputs;
  }
  
  // for debugging
  private void printMatrices(Matrix ma[]) {
    for (int i = 0; i < ma.length; i++) {
      System.out.println("matrix " + i);
      printMatrix(ma[i]);
    }
  }
  
  private void printMatrix(Matrix m) {
    for (int j = 0; j < m.numRows(); j++) {
      System.out.println("col " + j + ":" + m.viewRow(j));
    }
  }
  
  /**
   * 
   * @param output
   * @param target
   * @return cost function value
   */
  public double getCost(Vector output, Vector target){
    return costFunction.getCost(output, target);
  }
  
  @Override
  public Vector classifyFull(Vector instance) {
    //setUnitsWithForwardPropagation(instance);
    return forwardPropagation(instance);
  }
  
  @Override
  public Vector classify(Vector instance) {
    int nbOutput = nbOutputs;
    if (mutuallyExclusiveClasses) {
      Vector out = new DenseVector(nbOutput - 1);
      // assumes that outputs sum to 1:
      Vector outputUnits = forwardPropagation(instance);
      for (int i = 0; i < nbOutput - 1; i++) {
        out.setQuick(i, outputUnits.get(i));
      }
    } else {
      if (nbOutput == 1) {
        return forwardPropagation(instance);
      } else throw new UnsupportedOperationException();
    }
    throw new UnsupportedOperationException();
  }
  
  @Override
  public Vector classifyNoLink(Vector instance) {
    //
    return forwardPropagationNoLink(instance);
  }
  
  @Override
  public double classifyScalar(Vector instance) {
    Vector output = classifyFull(instance);
    return output.get(0);
  }
  
  public void copyFrom(MultiLayerPerceptron other) {
    
    learningRate = other.learningRate;
    regularization = other.regularization;
    momentum = other.momentum;
    mutuallyExclusiveClasses = other.mutuallyExclusiveClasses;
    hasNaturalPairing = other.hasNaturalPairing;
    for (int i = 0; i < nbLayer - 1; i++) {
      weights[i] = other.weights[i].clone();
    }
    for (int i = 0; i < nbLayer; i++) {
      squashingFunctions[i] = (Squashing) other.squashingFunctions[i];
    }
    costFunction = other.costFunction;
    
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(WRITABLE_VERSION);
    out.writeDouble(learningRate);
    out.writeDouble(regularization);
    out.writeDouble(momentum);
    out.writeBoolean(mutuallyExclusiveClasses);
    out.writeBoolean(hasNaturalPairing);
    out.writeInt(nbLayer);
    for (int i = 0; i < nbLayer - 1; i++) {
      MatrixWritable.writeMatrix(out, weights[i]);
    }
    for (int i = 0; i < nbLayer; i++) {
      out.writeUTF(squashingFunctions[i].name());
    }
    out.writeUTF(costFunction.name());
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    int version = in.readInt();
    if (version == WRITABLE_VERSION) {
      learningRate = in.readDouble();
      regularization = in.readDouble();
      momentum = in.readDouble();
      mutuallyExclusiveClasses = in.readBoolean();
      hasNaturalPairing = in.readBoolean();
      nbLayer = in.readInt();
      for (int i = 0; i < nbLayer - 1; i++) {
        weights[i] = MatrixWritable.readMatrix(in);
      }
      for (int i = 0; i < nbLayer; i++) {
        String squashingString = in.readUTF();
        squashingFunctions[i] = Squashing.valueOf(squashingString);
      }
      String costString = in.readUTF();
      costFunction = CostFunction.valueOf(costString);
      // initialize the rest from the information above
      initResidue();
    } else {
      throw new IOException("Incorrect object version, wanted "
          + WRITABLE_VERSION + " got " + version);
    }
  }
  
  @Override
  public void close() {
    // At moment this is an online classifier, nothing to do.
    // For batch learning: TODO
  }
  
  @Override
  public void train(long trackingKey, String groupKey, int actual,
      Vector instance) {
    // training with one pattern
    Vector target = new DenseVector(1);
    target.setQuick(0, (double) actual);
    trainOnline(instance, target);
  }
  
  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    train(trackingKey, null, actual, instance);
  }
  
  @Override
  public void train(int actual, Vector instance) {
    train(0, null, actual, instance);
  }

 
  
    
}



