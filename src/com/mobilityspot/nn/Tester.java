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
		
		IqNeuralNetwork myNN = new IqNeuralNetwork(inputs,outputs,dimensionsOfHiddenLayers,1000,IqNeuralNetworkMath.IRPROPMIN,true);

 
		
		myNN.trainNeuralNetwork();
 
		 /*
		 
		myNN.saveNNStatus("xor2.ser");
	 
		double[][] input_1_0 = {{1.0,0.0}};
		myNN.getNetworkOutput(input_1_0[0]);
		myNN.printNNStructure("test");
		
		double[][] input_0_1 = {{0.0,1.0}};
		myNN.getNetworkOutput(input_0_1[0]);
		myNN.printNNStructure("test");
		
		double[][] input_1_1 = {{1.0,1.0}};
		myNN.getNetworkOutput(input_1_1[0]);
		myNN.printNNStructure("test");
		
		double[][] input_0_0 = {{0.0,0.0}};
		myNN.getNetworkOutput(input_0_0[0]);
		myNN.printNNStructure("test");
		
		
		myNN.retriveNNStatusFromFile("xor2.ser").getNetworkOutput(input_1_0[0]);
		myNN.retriveNNStatusFromFile("xor2.ser").getNetworkOutput(input_0_1[0]);
		myNN.retriveNNStatusFromFile("xor2.ser").getNetworkOutput(input_1_1[0]);
		myNN.retriveNNStatusFromFile("xor2.ser").getNetworkOutput(input_0_0[0]);
 
		  */
		
	}

}
