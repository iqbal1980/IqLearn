package com.mobilityspot.nn;

import java.util.LinkedList;

public class Tester {

	public Tester() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		IQLayer ll0 = new IQLayer(2);
		IQLayer ll2 = new IQLayer(2);
		IQLayer ll3 = new IQLayer(1);
		double[][] inputs = {{1,0},{0,1},{1,1},{0,0}};
		double[][] outputs =  {{1},{1},{1},{0}};
		IQNeuralNetwork myNN = new IQNeuralNetwork(inputs,outputs);
		myNN.addLayer(ll0);

		myNN.addLayer(ll2);
		myNN.addLayer(ll3);
		
		
		myNN.printNNStructure();
		myNN.trainNeuralNetwork();
		myNN.printNNStructure();

		 
		
	}

}
