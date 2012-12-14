package com.mobilityspot.nn;

public class NNMAth implements java.io.Serializable {
	
	static double LEARNING_RATE =  1;
	static double LEARNING_MOMENTUM = 1;
	
	static double sigmoid(double input) {
		return (1 / (1 + Math.exp(-input) ) );
	}
	
	static double sigmoidDerivative(double input) {
		return (  Math.exp(input) / ((Math.exp(input) +1)*(Math.exp(input) +1))  ) ;
	}
	
	static double errorThreshHold() {
		return 0.0005;
	}
 
}
