package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hama.ml.math.DenseDoubleMatrix;
import org.apache.hama.ml.math.DoubleMatrix;
import org.apache.hama.ml.writable.MatrixWritable;

/**
 * SmallMLPMessage is used to exchange information for the
 * {@link SmallMultiLayerPerceptron}.
 * It send the whole parameter matrix from one task to another.
 * 
 */
public class SmallMLPMessage extends MLPMessage {
	
	private int owner;	//	the ID of the task who creates the message
	private DenseDoubleMatrix updatedWeights;	
	
	public SmallMLPMessage() {
	}
	
	public SmallMLPMessage(int owner, boolean terminated, DenseDoubleMatrix mat) {
		super(terminated);
		this.owner = owner;
		this.updatedWeights = mat;
	}

	/**
	 * Get the owner task Id of the message.
	 * @return
	 */
	public int getOwner() {
		return owner;
	}
	
	/**
	 * Get the updated weights.
	 * @return
	 */
	public DenseDoubleMatrix getUpdatedWeights() {
		return this.updatedWeights;
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		this.owner = input.readInt();
		this.terminated = input.readBoolean();
		this.updatedWeights = (DenseDoubleMatrix)MatrixWritable.read(input);
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeInt(this.owner);
		output.writeBoolean(this.terminated);
		MatrixWritable.write(this.updatedWeights, output);
	}

}
