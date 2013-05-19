package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.writable.MatrixWritable;

/**
 * SmallMLPMessage is used to exchange information for the
 * {@link SmallMultiLayerPerceptron}.
 * It send the whole parameter matrix from one task to another.
 * 
 */
public class SmallMLPMessage implements MLPMessage {
	
	private MatrixWritable matrix;
	
	public SmallMLPMessage(DoubleMatrix mat) {
		this.matrix = new MatrixWritable(mat);
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		this.matrix.readFields(input);
	}

	@Override
	public void write(DataOutput output) throws IOException {
		this.matrix.write(output);
	}

}
