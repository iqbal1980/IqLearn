package com.mobilityspot.nn;

public class NNMAth {

	static double sigmoid(double input) {
		return (1 / (1 + Math.exp(-input) ) );
	}
	
	static double sigmoidDerivative(double input) {
		return (  Math.exp(input) / ((Math.exp(input) +1)*(Math.exp(input) +1))  ) ;
	}
 
}