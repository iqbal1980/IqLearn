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
		double[][] outputs =  {{1.0},{1.0},{0.0},{0.0}};
		
		IqNeuralNetwork myNN = new IqNeuralNetwork(inputs,outputs,dimensionsOfHiddenLayers,1500,IqNeuralNetworkMath.IRPROPMIN,true);

 
		 
		myNN.trainNeuralNetwork();
 
		
 
		myNN.saveNNStatus("xor2.ser");
	 /*
		
	  double[][] toto = {{1.0,1.0}};
		 
		 myNN.getNetworkOutput(toto[0]);
		 myNN.printNNStructure("tesT");
	 
	  
	  myNN.retriveNNStatusFromFile("xor2.ser").getNetworkOutput(toto[0]);
 */
		 
		
	}

}
