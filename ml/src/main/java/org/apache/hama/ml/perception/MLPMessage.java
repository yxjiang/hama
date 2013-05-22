package org.apache.hama.ml.perception;

import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

/**
 * MLPMessage is used to hold the parameters that needs
 * to be sent between the tasks.
 *
 */
public abstract class MLPMessage implements Writable {
	protected BooleanWritable terminated;
	
	public void setTerminated(boolean terminated) {
		this.terminated = new BooleanWritable(terminated);
	}

	public boolean isTerminated() {
		return terminated.get();
	}
	
	
	
}
