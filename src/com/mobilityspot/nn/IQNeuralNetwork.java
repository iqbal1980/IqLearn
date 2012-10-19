package com.mobilityspot.nn;

import java.util.ArrayList;
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
			
			//for(int i = 0; i < 10000 ; i++) { //10000 = training iterations
				
				for(int j = 0; j < this.inputs.length; j++) {
					
					System.out.println(getNetworkOutput(inputs[j])[0]);
					
				}
				
			//}	
			
		}
	}
	
	public double[]  getNetworkOutput(double[] inputsValues) {
		layers.getFirst().updateNodesValues(inputsValues);
		double newValue = layers.getFirst().getLayerOutput();

		for(int i = 1; i < layers.size() - 1; i++) {
			layers.get(i).updateNodesValues(newValue);
			newValue = layers.get(i).getLayerOutput();
		}
		
		
		layers.getLast().updateNodesValues(newValue);
		
		/*for(IQNode myNode : layers.getLast().getNodes()) {
			myNode.setValue(newValue);
		}*/
		
		int j = 0;
		double[] returnLastLayerValues = new double[expectedOutputs.length];

		for(IQNode myNode2 : layers.getLast().getNodes()) {
			returnLastLayerValues[j] = myNode2.getValue() * myNode2.getWeight();
			j++;
		}
		
		return returnLastLayerValues;
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
