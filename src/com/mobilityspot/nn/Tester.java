package com.mobilityspot.nn;


public class Tester {

	public Tester() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

		int[] dimensionsOfHiddenLayers = {3};
		double[][] inputs = {{1,0},{0,1},{1,1},{0,0}};
		double[][] outputs =  {{1},{1},{0},{0}};
		
		IqNeuralNetwork myNN = new IqNeuralNetwork(inputs,outputs,dimensionsOfHiddenLayers);
 
		
		 
		myNN.trainNeuralNetwork();
	  double[][] toto = {{0.0001,0.00002}};
		myNN.getNetworkOutput(toto[0]);
		
		myNN.printNNStructure();

		 
		
	}

}
