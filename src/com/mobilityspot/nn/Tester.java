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

		int[] dimensionsOfHiddenLayers = {2};
		double[][] inputs = {{1,0},{0,1},{1,1},{0,0}};
		double[][] outputs =  {{1},{1},{1},{0}};
		
		IQNeuralNetwork myNN = new IQNeuralNetwork(inputs,outputs,dimensionsOfHiddenLayers);
 
		
		//myNN.printNNStructure();
		myNN.trainNeuralNetwork();
		myNN.printNNStructure();
 

		 
		
	}

}
