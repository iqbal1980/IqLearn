package com.mobilityspot.nn;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

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
	public IQNeuralNetwork(double[][] myInputs, double[][] myExpectedOutputs,  int[] dimensionsOfHiddenLayers) throws Exception {
		this.inputs = myInputs;
		this.expectedOutputs = myExpectedOutputs;
		
		//System.out.println(myInputs[0].length);
		
		int numberOfInputs = myInputs[0].length;
		int numberOfOutpus = myExpectedOutputs[0].length;
		
		IQLayer inputLayer = new IQLayer(numberOfInputs);
		IQLayer outputLayer = new IQLayer(numberOfOutpus);
		
		layers.add(inputLayer);
		

		for(int i = 0 ; i < dimensionsOfHiddenLayers.length ; i++) {
				IQLayer inputLayerTmp = null;
				inputLayerTmp = new IQLayer(dimensionsOfHiddenLayers[i]);
				layers.add(inputLayerTmp);
		}
		
		layers.add(outputLayer);
		
		
		for(IQNode myNode : layers.getFirst().getNodes()) {
			myNode.setWeights(null);
		}
		
		for(int i=0;i<layers.size();i++) {
			if(layers.listIterator(i).hasPrevious() == true) {
				int previousLayerSize = layers.listIterator(i).previous().getNodes().size();
				int currentLayerSize =  layers.get(i).getNodes().size();
				for(IQNode myNode : layers.get(i).getNodes()) {
					myNode.initNodeWeights(previousLayerSize);
				}
			}
		}
		
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
	
	
	
	public void resetNeuralNetwork() {
		
	}
	
	public void trainNeuralNetwork() throws Exception {
		//displayInputs();
		//displayOutputs();
		//printNNStructure();
		double[] netWorkOutPut = null;
		for(int i = 0; i < 1  ; i++) { //training iterations
			for(int j = 0; j<1/*inputs.length*/;j++) {
				netWorkOutPut = getNetworkOutput(inputs[j]);
				backPropagateError(netWorkOutPut,this.expectedOutputs[j]);
			}
		}
	}
	
	public double[]  getNetworkOutput(double[] inputsValues) {
		for(int i=0 ;i < layers.getFirst().getNodes().size(); i++) {
			layers.getFirst().getNodes().get(i).setValue(inputsValues[i]);
			System.out.println(inputsValues[i]);
		}
		
		for(int i=0; i<layers.size(); i++) {
			if(layers.listIterator(i).hasPrevious() == true) {
				int previousLayerSize = layers.listIterator(i).previous().getNodes().size();
				int currentLayerSize =  layers.get(i).getNodes().size();

				for(IQNode currentLayerNode : layers.get(i).getNodes()) {
					double newValue = 0;
					for(int j=0 ; j < currentLayerNode.getWeights().size();j++) {
						newValue += currentLayerNode.getWeights().get(j) * layers.listIterator(i).previous().getNodes().get(j).getValue();
					}
					currentLayerNode.setValue(NNMAth.sigmoid(newValue));
					
				}
			}
		}
		
		double[] returnValue = new double[layers.getLast().getNodes().size()];
		for(int i = 0 ; i< layers.getLast().getNodes().size(); i++) {
			returnValue[i] = layers.getLast().getNodes().get(i).getValue();
		}
	 
		return returnValue;

	}
	
	
	public void backPropagateError(double[] nnOutput, double[] nnExpectedOutputs) {
		
		for(int i=0 ;i < layers.getLast().getNodes().size(); i++) {
			double currentErrorDelta = 0;
			currentErrorDelta = layers.getLast().getNodes().get(i).getValue();
			currentErrorDelta = (1 - currentErrorDelta)*(nnOutput[i] - currentErrorDelta);
			
			ArrayList<Double> updatedWeights = new ArrayList<Double>(layers.getLast().getNodes().get(i).getWeights().size());
			
			for(int j = 0 ; j < layers.getLast().getNodes().get(i).getWeights().size(); j++) {
				double myWeight = 0;
				myWeight = layers.getLast().getNodes().get(i).getWeights().get(j).doubleValue();
				myWeight = myWeight + NNMAth.LEARNING_RATE * (layers.getLast().getNodes().get(i).getError()) * (layers.listIterator(i).previous().getNodes().get(j).getValue());
				updatedWeights.add(myWeight);
			}
			
			layers.getLast().getNodes().get(i).getWeights().clear();
			layers.getLast().getNodes().get(i).setWeights(updatedWeights);
		}
		
		
		for(int k = (layers.size() - 1); k > 1 ; k--) {
			if(layers.listIterator(k).hasNext() == true) {
				
				for(int l=0 ;l < layers.get(k).getNodes().size(); l++) {
					double currentErrorDelta = 0;
					currentErrorDelta = layers.get(k).getNodes().get(l).getValue();
					currentErrorDelta *= (1 - currentErrorDelta);
					
					double nextLayerErrorAndWeightsFactor = 0;
					
					for(int m = 0 ; m < layers.listIterator(k).next().getNodes().size(); m++) {
						nextLayerErrorAndWeightsFactor += layers.listIterator(k).next().getNodes().get(m).getError() * layers.listIterator(k).next().getNodes().get(m).getWeights().get(l).doubleValue();	 
					}
					
					currentErrorDelta *= nextLayerErrorAndWeightsFactor;
				}
				
			}
		}

		
		
		for(int n=0 ;n < layers.getFirst().getNodes().size(); n++) {
			double currentErrorDelta = 0;
			
		}
		
		
		
	}
	

	public void printNNStructure() {
		System.out.println("===================================================================================================================");
		for(IQLayer layer : layers) {
			System.out.println("layer **************"+ layers.indexOf(layer));
			for(IQNode node : layer.getNodes()) {
				System.out.println("node number " + layer.getNodes().indexOf(node));
				System.out.println("node  value " + node.getValue());
				System.out.println("layerWeights = ");  
				if(node.getWeights() != null) {
					int counter = 0;
					for(Double currentWeight : node.getWeights()) {
						System.out.println("layerWeight "+counter +" =|= "+currentWeight);
						counter++;
					}
				}

			}
			//System.out.println("layer output = " + layer.getLayerOutput());
		}
		System.out.println("===================================================================================================================");
	}
	
}
