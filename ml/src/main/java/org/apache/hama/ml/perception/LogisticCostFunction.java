package org.apache.hama.ml.perception;

public class LogisticCostFunction extends CostFunction {
	
	@Override
	public double getCost(double target, double actual) {
		return - target * Math.log(actual) - (1 - target) * Math.log(1 - actual);
	}

	@Override
	public double getPartialDerivative(double target, double actual) {
		if (actual == 1) {
			actual = 0.999;
		}
		else if (actual == 0) {
			actual = 0.001;
		}
		if (target == 1) {
			target = 0.999;
		}
		else if (target == 0) {
			target = 0.001;
		}
		return - target / actual + (1 - target) / (1 - actual);
	}

}
