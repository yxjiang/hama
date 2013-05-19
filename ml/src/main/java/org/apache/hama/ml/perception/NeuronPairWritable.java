package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * The NeuronPairWritable is used to store the weight between to neurons.
 *
 */
public class NeuronPairWritable implements Writable {
	
	private long firstNeuronId;
	private long secondNeuronId;
	private double weight;

	@Override
	public void readFields(DataInput input) throws IOException {
		this.firstNeuronId = input.readLong();
		this.secondNeuronId = input.readLong();
		this.weight = input.readDouble();
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeLong(firstNeuronId);
		output.writeLong(secondNeuronId);
		output.writeDouble(weight);
	}

}
