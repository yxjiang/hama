package org.apache.hama.ml.perception;

import org.apache.hama.ml.math.DoubleVector;

/**
 * The classificationPerceptron is used to conduct the classification task.
 * From the perspective of topology, it is a multilayer perceptron that consists
 * of one input layer, multiple hidden layer and one output layer.
 * 
 * The number of neurons in the input layer should be consistent with the
 * number of features in the training instance.
 * The number of neurons in the output layer
 *
 */
public interface ClassificationPerceptron extends PerceptronBase {

	/**
	 * Classify the input instance.
	 * @param featureVector
	 * @return	The classification results.
	 */
	public abstract DoubleVector classify(DoubleVector featureVector);
}
