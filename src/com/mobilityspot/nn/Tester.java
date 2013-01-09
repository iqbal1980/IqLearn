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
		
		IqNeuralNetwork myNN = new IqNeuralNetwork(inputs,outputs,dimensionsOfHiddenLayers,1 );

 
		 
		myNN.trainNeuralNetwork();
		
		/*
		myNN.saveNNStatus("xor2.ser");
	 
		
	  double[][] toto = {{0.0,0.0}};
		 
		 myNN.getNetworkOutput(toto[0]);
		 myNN.printNNStructure();
	 
	  
	  myNN.retriveNNStatusFromFile("xor2.ser").getNetworkOutput(toto[0]);
 */
		 
		
	}

}
