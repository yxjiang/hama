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
package org.apache.hama.commons.math;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.google.common.base.Preconditions;

/**
 * @author yjian004
 * 
 */
public class SparseDoubleVector implements DoubleVector {

  private int dimension;
  private double defaultValue; // 0 by default
  private Map<Integer, Double> elements;

  public SparseDoubleVector(int dimension) {
    this(dimension, 0.0);
  }

  private SparseDoubleVector(int dimension, double defaultValue) {
    this.elements = new HashMap<Integer, Double>();
    this.defaultValue = defaultValue;
    this.dimension = dimension;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#get(int)
   */
  @Override
  public double get(int index) {
    Preconditions.checkArgument(index < this.dimension,
        "Index out of max allowd dimension of sparse vector.");
    Double val = this.elements.get(index);
    if (val == null) {
      val = this.defaultValue;
    }
    return val;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#getLength()
   */
  @Override
  public int getLength() {
    return this.dimension;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#getDimension()
   */
  @Override
  public int getDimension() {
    return this.dimension;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#set(int, double)
   */
  @Override
  public void set(int index, double value) {
    Preconditions.checkArgument(index < this.dimension,
        "Index out of max allowd dimension of sparse vector.");
    this.elements.put(index, value);
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#applyToElements(org.apache.hama
   * .commons.math.DoubleFunction)
   */
  @Override
  public DoubleVector applyToElements(DoubleFunction func) {
    SparseDoubleVector newVec = new SparseDoubleVector(this.dimension,
        func.apply(this.defaultValue));
    // apply function to all non-empty entries
    for (Map.Entry<Integer, Double> entry : this.elements.entrySet()) {
      newVec.elements.put(entry.getKey(), func.apply(entry.getValue()));
    }

    return newVec;
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#applyToElements(org.apache.hama
   * .commons.math.DoubleVector,
   * org.apache.hama.commons.math.DoubleDoubleFunction)
   */
  @Override
  public DoubleVector applyToElements(DoubleVector other,
      DoubleDoubleFunction func) {
    SparseDoubleVector newVec = new SparseDoubleVector(this.dimension,
        this.defaultValue);

    Iterator<DoubleVectorElement> otherItr = other.iterate();
    while (otherItr.hasNext()) {
      DoubleVectorElement element = otherItr.next();
      int index = element.getIndex();
      double otherVal = element.getValue();
      Double val = this.elements.get(index);
      if (val == null) { // use default value
        double resVal = func.apply(newVec.defaultValue, otherVal);
        if (resVal != this.defaultValue) {
          newVec.elements.put(index, resVal);
        }
      } else {
        newVec.elements.put(index, func.apply(val, otherVal));
      }

    }

    return newVec;
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#addUnsafe(org.apache.hama.commons
   * .math.DoubleVector)
   */
  @Override
  public DoubleVector addUnsafe(DoubleVector vector) {
    return this.applyToElements(vector, new DoubleDoubleFunction() {
      @Override
      public double apply(double x1, double x2) {
        return x1 + x2;
      }

      @Override
      public double applyDerivative(double x1, double x2) {
        throw new UnsupportedOperationException();
      }
    });
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#add(org.apache.hama.commons.math
   * .DoubleVector)
   */
  @Override
  public DoubleVector add(DoubleVector vector) {
    Preconditions.checkArgument(this.dimension == vector.getDimension(),
        "Dimensions of two vectors are not the same.");
    return this.addUnsafe(vector);
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#add(double)
   */
  @Override
  public DoubleVector add(double scalar) {
    final double val = scalar;
    return this.applyToElements(new DoubleFunction() {
      @Override
      public double apply(double value) {
        return value + val;
      }

      @Override
      public double applyDerivative(double value) {
        throw new UnsupportedOperationException();
      }
    });
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#subtractUnsafe(org.apache.hama
   * .commons.math.DoubleVector)
   */
  @Override
  public DoubleVector subtractUnsafe(DoubleVector vector) {
    return this.applyToElements(vector, new DoubleDoubleFunction() {
      @Override
      public double apply(double x1, double x2) {
        return x1 - x2;
      }

      @Override
      public double applyDerivative(double x1, double x2) {
        return 0;
      }
    });
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#subtract(org.apache.hama.commons
   * .math.DoubleVector)
   */
  @Override
  public DoubleVector subtract(DoubleVector vector) {
    Preconditions.checkArgument(this.dimension == vector.getDimension(),
        "Dimensions of two vector are not the same.");
    return this.subtractUnsafe(vector);
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#subtract(double)
   */
  @Override
  public DoubleVector subtract(double scalar) {
    final double val = scalar;
    return this.applyToElements(new DoubleFunction() {
      @Override
      public double apply(double value) {
        return value - val;
      }

      @Override
      public double applyDerivative(double value) {
        throw new UnsupportedOperationException();
      }
    });
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#subtractFrom(double)
   */
  @Override
  public DoubleVector subtractFrom(double scalar) {
    final double val = scalar;
    return this.applyToElements(new DoubleFunction() {
      @Override
      public double apply(double value) {
        return val - value;
      }

      @Override
      public double applyDerivative(double value) {
        throw new UnsupportedOperationException();
      }
    });
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#multiply(double)
   */
  @Override
  public DoubleVector multiply(double scalar) {
    final double val = scalar;
    return this.applyToElements(new DoubleFunction() {
      @Override
      public double apply(double value) {
        return val * value;
      }

      @Override
      public double applyDerivative(double value) {
        throw new UnsupportedOperationException();
      }
    });
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#multiplyUnsafe(org.apache.hama
   * .commons.math.DoubleVector)
   */
  @Override
  public DoubleVector multiplyUnsafe(DoubleVector vector) {
    return this.applyToElements(vector, new DoubleDoubleFunction() {
      @Override
      public double apply(double x1, double x2) {
        return x1 * x2;
      }

      @Override
      public double applyDerivative(double x1, double x2) {
        throw new UnsupportedOperationException();
      }
    });
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#multiply(org.apache.hama.commons
   * .math.DoubleVector)
   */
  @Override
  public DoubleVector multiply(DoubleVector vector) {
    Preconditions.checkArgument(this.dimension == vector.getDimension(),
        "Dimensions of two vectors are not the same.");
    return this.multiplyUnsafe(vector);
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#multiply(org.apache.hama.commons
   * .math.DoubleMatrix)
   */
  @Override
  public DoubleVector multiply(DoubleMatrix matrix) {
    Preconditions
        .checkArgument(this.dimension == matrix.getColumnCount(),
            "The dimension of vector does not equal to the dimension of the matrix column.");
    return this.multiplyUnsafe(matrix);
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#multiplyUnsafe(org.apache.hama
   * .commons.math.DoubleMatrix)
   */
  @Override
  public DoubleVector multiplyUnsafe(DoubleMatrix matrix) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#divide(double)
   */
  @Override
  public DoubleVector divide(double scalar) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#divideFrom(double)
   */
  @Override
  public DoubleVector divideFrom(double scalar) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#pow(int)
   */
  @Override
  public DoubleVector pow(int x) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#abs()
   */
  @Override
  public DoubleVector abs() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#sqrt()
   */
  @Override
  public DoubleVector sqrt() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#sum()
   */
  @Override
  public double sum() {
    // TODO Auto-generated method stub
    return 0;
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#dotUnsafe(org.apache.hama.commons
   * .math.DoubleVector)
   */
  @Override
  public double dotUnsafe(DoubleVector vector) {
    // TODO Auto-generated method stub
    return 0;
  }

  /*
   * (non-Javadoc)
   * @see
   * org.apache.hama.commons.math.DoubleVector#dot(org.apache.hama.commons.math
   * .DoubleVector)
   */
  @Override
  public double dot(DoubleVector vector) {
    // TODO Auto-generated method stub
    return 0;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#slice(int)
   */
  @Override
  public DoubleVector slice(int length) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#sliceUnsafe(int)
   */
  @Override
  public DoubleVector sliceUnsafe(int length) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#slice(int, int)
   */
  @Override
  public DoubleVector slice(int start, int end) {
    Preconditions.checkArgument(start >= 0 && end < this.dimension, String
        .format("Start and end range should be in [0, %d].", this.dimension));
    return this.sliceUnsafe(start, end);
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#sliceUnsafe(int, int)
   */
  @Override
  public DoubleVector sliceUnsafe(int start, int end) {
    SparseDoubleVector slicedVec = new SparseDoubleVector(end - start);
    slicedVec.elements = new HashMap<Integer, Double>();
    for (Map.Entry<Integer, Double> entry : this.elements.entrySet()) {
      if (entry.getKey() >= start && entry.getKey() <= end) {
        slicedVec.elements.put(entry.getKey(), entry.getValue());
      }
    }
    return slicedVec;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#max()
   */
  @Override
  public double max() {
    double max = this.defaultValue;
    for (Map.Entry<Integer, Double> entry : this.elements.entrySet()) {
      max = Math.max(max, entry.getValue());
    }
    return max;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#min()
   */
  @Override
  public double min() {
    double min = this.defaultValue;
    for (Map.Entry<Integer, Double> entry : this.elements.entrySet()) {
      min = Math.min(min, entry.getValue());
    }
    return min;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#toArray()
   */
  @Override
  public double[] toArray() {
    throw new UnsupportedOperationException(
        "SparseDoubleVector does not support toArray() method.");
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#deepCopy()
   */
  @Override
  public DoubleVector deepCopy() {
    SparseDoubleVector copy = new SparseDoubleVector(this.dimension);
    copy.elements = new HashMap<Integer, Double>(this.elements.size());
    copy.elements.putAll(this.elements);
    return copy;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#iterateNonZero()
   */
  @Override
  public Iterator<DoubleVectorElement> iterateNonZero() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#iterate()
   */
  @Override
  public Iterator<DoubleVectorElement> iterate() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#isSparse()
   */
  @Override
  public boolean isSparse() {
    return true;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#isNamed()
   */
  @Override
  public boolean isNamed() {
    return false;
  }

  /*
   * (non-Javadoc)
   * @see org.apache.hama.commons.math.DoubleVector#getName()
   */
  @Override
  public String getName() {
    return null;
  }

}