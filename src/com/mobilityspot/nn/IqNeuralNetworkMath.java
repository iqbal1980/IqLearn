package com.mobilityspot.nn;

public class IqNeuralNetworkMath implements java.io.Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8433685280777325575L;
	static double LEARNING_RATE =  1;
	static double LEARNING_MOMENTUM = 1;
	static int BACKPROP = 0;
	static int BATCHBACKPROP = 1;
	static int IRPROPMIN = 2;
	
	static double sigmoid(double input) {
		return (1 / (1 + Math.exp(-input) ) );
	}
	
	static double sigmoidDerivative(double input) {
		return (  Math.exp(input) / ((Math.exp(input) +1)*(Math.exp(input) +1))  ) ;
	}
	
	static double errorThreshHold() {
		return 0.0005;
	}
 
	
    static int sign(final double value) {
        if (Math.abs(value) < 0.00000000000000001) {
                return 0;
        } else if (value > 0) {
                return 1;
        } else {
                return -1;
        }
    }
}
