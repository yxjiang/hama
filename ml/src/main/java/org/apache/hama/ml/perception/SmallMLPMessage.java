package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.writable.MatrixWritable;

/**
 * SmallMLPMessage is used to exchange information for the
 * {@link SmallMultiLayerPerceptron}.
 * It send the whole parameter matrix from one task to another.
 * 
 */
public class SmallMLPMessage extends MLPMessage {
	
	private IntWritable owner;	//	the ID of the task who creates the message
	private MatrixWritable matrix;
	
	public SmallMLPMessage(int owner, DoubleMatrix mat) {
		this.owner = new IntWritable(owner);
		this.matrix = new MatrixWritable(mat);
	}

	/**
	 * Get the owner task Id of the message.
	 * @return
	 */
	public IntWritable getOwner() {
		return owner;
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		this.owner.readFields(input);
		this.terminated.readFields(input);
		this.matrix.readFields(input);
	}

	@Override
	public void write(DataOutput output) throws IOException {
		this.owner.write(output);
		this.terminated.write(output);
		this.matrix.write(output);
	}

}
