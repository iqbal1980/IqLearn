package com.mobilityspot.nn;

import java.util.LinkedList;

public class IQNeuralNetwork {
	private LinkedList<IQLayer> layers  = new LinkedList<IQLayer>();
	private double[][] inputs;
	private double[][] expectedOutputs;

	
	public void displayInputs() {
		System.out.println("**************************************");
		for(int i=0;i < this.inputs.length ; i++) {
			for(int j=0;j < this.inputs[i].length ; j++) {
				System.out.println(this.inputs[i][j]);
			}
		}
		System.out.println("**************************************");
	}
	
	public void displayOutputs() {
		System.out.println("**************************************");
		for(int i=0;i < this.expectedOutputs.length ; i++) {
			for(int j=0;j < this.expectedOutputs[i].length ; j++) {
				System.out.println(this.expectedOutputs[i][j]);
			}
		}
		System.out.println("**************************************");
	}
	public IQNeuralNetwork(double[][] myInputs, double[][] myExpectedOutputs) {
		this.inputs = myInputs;
		this.expectedOutputs = myExpectedOutputs;
	}
	
	public IQNeuralNetwork(LinkedList<IQLayer> nnLayers , double[][] myInputs, double[][] myExpectedOutputs) {
		this.layers = nnLayers;
		this.inputs = myInputs;
		this.expectedOutputs = myExpectedOutputs;
	}

	public LinkedList<IQLayer> getLayers() {
		return layers;
	}

	public void setLayers(LinkedList<IQLayer> layers) {
		this.layers = layers;
	}
	
	public void addLayer(IQLayer layer) {
		this.layers.add(layer);
	}
	
	
	public void trainNeuralNetwork() throws Exception {
		//displayInputs();
		//displayOutputs();
 
		if(this.inputs[0].length != layers.getFirst().getNodes().size() || this.expectedOutputs[0].length != layers.getLast().getNodes().size()) {
			throw new Exception("Input or output size not matching first and/or last layers");
		} else {
			
			for(int i = 0; i < 1000 ; i++) { //100 = training iterations
				
				for(int j = 0; j < this.inputs.length; j++) {
					/*for(IQNode node : layers.getFirst().getNodes()) {
						node.setValue(this.input[0]);
					}*/
				}
				
			}
			
			
			
		}
	}
	
	public void printNNStructure() {
		for(IQLayer layer : layers) {
			System.out.println("layer **************"+ layers.indexOf(layer));
			for(IQNode node : layer.getNodes()) {
				System.out.println("node number " + layer.getNodes().indexOf(node));
				System.out.println("node weight = " + node.getWeight() + " & node value = " + node.getValue());
			}
			System.out.println("layer output = " + layer.getLayerOutput());
		}
	}
	
}
