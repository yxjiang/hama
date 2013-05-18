package org.apache.hama.ml.perception;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hama.ml.math.DenseDoubleMatrix;

/**
 * SmallMLPMessage is used to exchange information for the
 * {@link SmallMultiLayerPerceptronBSP}.
 * It send the whole parameter matrix from one task to another.
 *
 */
public class SmallMLPMessage implements MLPMessage {
	
	private DenseDoubleMatrix matrix;

	@Override
	public void readFields(DataInput arg0) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void write(DataOutput arg0) throws IOException {
		// TODO Auto-generated method stub
		
	}


}
