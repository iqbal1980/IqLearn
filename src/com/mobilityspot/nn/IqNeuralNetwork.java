package com.mobilityspot.nn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.LinkedList;

public class IqNeuralNetwork implements java.io.Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1488930130307912669L;
	private LinkedList<IqLayer> layers  = new LinkedList<IqLayer>();
	private double[][] inputs;
	private double[][] expectedOutputs;

 
	
 
	
	
	public IqNeuralNetwork(double[][] myInputs, double[][] myExpectedOutputs,  int[] dimensionsOfHiddenLayers) throws Exception {
		this.inputs = myInputs;
		this.expectedOutputs = myExpectedOutputs;
		
		//System.out.println(myInputs[0].length);
		
		int numberOfInputs = myInputs[0].length;
		int numberOfOutpus = myExpectedOutputs[0].length;
		
		IqLayer inputLayer = new IqLayer(numberOfInputs,true);
		
		
		layers.addFirst(inputLayer);
		
		for(int i = 0 ; i < dimensionsOfHiddenLayers.length ; i++) {
			System.out.println("==============>>>>"+i);
				IqLayer inputLayerTmp = null;
				inputLayerTmp = new IqLayer(dimensionsOfHiddenLayers[i],true);
				layers.add(inputLayerTmp);
		}
		
		IqLayer outputLayer = new IqLayer(numberOfOutpus,true);
		layers.addLast(outputLayer);
		
		
		for(IqNode myNode : layers.getFirst().getNodes()) {
			myNode.setWeights(null);
		}
		
		for(int i=0;i<layers.size();i++) {
			System.out.println("i === "+i + " layers.listIterator(i).hasPrevious() === "+ layers.listIterator(i).hasPrevious());
			if(layers.listIterator(i).hasPrevious() == true) {
				int previousLayerSize = layers.get(i - 1).getNodes().size();//layers.listIterator(i).previous().getNodes().size();
				for(IqNode myNode : layers.get(i).getNodes()) {
					myNode.initNodeWeights(previousLayerSize);
				}
			}
		}
		
	}

	public LinkedList<IqLayer> getLayers() {
		return layers;
	}

	public void setLayers(LinkedList<IqLayer> layers) {
		this.layers = layers;
	}
	
 
	
	
	public void resetNeuralNetwork() {
		
	}
	
	public void trainNeuralNetwork() throws Exception {
 
		//printNNStructure();
		double[] netWorkOutPut = null;
		for(int i = 0; i < 10000  ; i++) { //training iterations
			for(int j = 0; j<  inputs.length    ;j++) {
				netWorkOutPut = getNetworkOutput(inputs[j]);
				//printNNStructure();
				backPropagateError(netWorkOutPut,this.expectedOutputs[j]);
				printNNStructure();
			}
		}
	}
	
	public double[]  getNetworkOutput(double[] inputsValues) {
		for(int i=0 ;i < layers.getFirst().getNodes().size() - 1; i++) {//-1 account for disregarding bias node
			layers.getFirst().getNodes().get(i).setValue(inputsValues[i]);
		}
		
		for(int i=0; i<layers.size(); i++) {
			if(layers.listIterator(i).hasPrevious() == true) {
				//int previousLayerSize = layers.get(i - 1).getNodes().size();//layers.listIterator(i).previous().getNodes().size();
				//int currentLayerSize =  layers.get(i).getNodes().size();

				for(IqNode currentLayerNode : layers.get(i).getNodes()) {
					double newValue = 0;
					for(int j=0 ; j < currentLayerNode.getWeights().size();j++) {
						newValue += currentLayerNode.getWeights().get(j) * layers.get(i - 1).getNodes().get(j).getValue();//layers.listIterator(i).previous().getNodes().get(j).getValue();
					}
					
					if(currentLayerNode.isBiasNode() == false) {
						currentLayerNode.setValue(IqNeuralNetworkMath.sigmoid(newValue));
					}
					
				}
			}
		}
		
		double[] returnValue = new double[layers.getLast().getNodes().size()];
		for(int i = 0 ; i< layers.getLast().getNodes().size(); i++) {
			returnValue[i] = layers.getLast().getNodes().get(i).getValue();
		}
		printNNStructure();
		return returnValue;

	}
	
	
	public void backPropagateError(double[] nnOutput, double[] nnExpectedOutputs) {
		for(int i=0 ;i < layers.getLast().getNodes().size() - 1; i++) {//-1 account for disregarding bias node
			double currentErrorDelta = 0;
			currentErrorDelta = layers.getLast().getNodes().get(i).getValue();
			currentErrorDelta = currentErrorDelta*(1 - currentErrorDelta)*(nnExpectedOutputs[i] - currentErrorDelta);
 
			
			
			layers.getLast().getNodes().get(i).setError(currentErrorDelta);
			
			ArrayList<Double> updatedWeights = new ArrayList<Double>(layers.getLast().getNodes().get(i).getWeights().size());
			
			for(int j = 0 ; j < layers.getLast().getNodes().get(i).getWeights().size(); j++) {
				double myWeight = 0;
				myWeight = layers.getLast().getNodes().get(i).getWeights().get(j).doubleValue();
 
				int layersSize = layers.size() - 1;

				myWeight = IqNeuralNetworkMath.LEARNING_MOMENTUM*myWeight + IqNeuralNetworkMath.LEARNING_RATE * (layers.getLast().getNodes().get(i).getError()) * layers.get(layersSize - 1).getNodes().get(j).getValue();//(layers.listIterator(layersSize).previous().getNodes().get(j).getValue());
				updatedWeights.add(myWeight);
			}
			
			layers.getLast().getNodes().get(i).getWeights().clear();
			layers.getLast().getNodes().get(i).setWeights(updatedWeights);	
		}

		
		 
		System.out.println("size of layers >>>>>>>>>>>>>>"+layers.size());
		
		for(int k = (layers.size() - 2 ); k >=0 ; k--) {//looping trough middle hidden layers
			if(layers.listIterator(k).hasPrevious() == true) {
				 for(int l = 0; l < layers.get(k).getNodes().size(); l++) {//looping trough nodes of layer
					 double deltaNodeError = 0;
					 deltaNodeError = layers.get(k).getNodes().get(l).getValue();
					 deltaNodeError *= 1 - deltaNodeError;
					 double newSigmaOfWxDeltaErrors = 0;
					 for(int m = 0; m < layers.get(k + 1).getNodes().size(); m++) {//looping trough next layer nodes
						 newSigmaOfWxDeltaErrors += layers.get(k + 1).getNodes().get(m).getWeights().get(l).doubleValue() * layers.get(k+1).getNodes().get(m).getError();
					 }
					 
					 deltaNodeError *= newSigmaOfWxDeltaErrors;
					 
					 layers.get(k).getNodes().get(l).setError(deltaNodeError);
					 
					 ArrayList<Double> newWeights = new ArrayList<Double>();
					 newWeights.clear();
					 for(int n = 0; n < layers.get(k).getNodes().get(l).getWeights().size(); n++) {
						 double newNodeWeight = IqNeuralNetworkMath.LEARNING_MOMENTUM*layers.get(k).getNodes().get(l).getWeights().get(n).doubleValue() + IqNeuralNetworkMath.LEARNING_RATE*layers.get(k).getNodes().get(l).getError()*layers.get(k-1).getNodes().get(n).getValue();
						 newWeights.add(newNodeWeight);
					 }
					 layers.get(k).getNodes().get(l).getWeights().clear();
					 layers.get(k).getNodes().get(l).setWeights(newWeights);
				 }
			}
		}
		
		
 

		
		
		
	}
	

	public void printNNStructure() {
		System.out.println("===================================================================================================================");
		for(IqLayer layer : layers) {
			System.out.println("layer **************"+ layers.indexOf(layer));
			for(IqNode node : layer.getNodes()) {
				if(layers.indexOf(layer) == 0 || layers.indexOf(layer) == (layers.size() - 1) ) {
					System.out.println("node number " + layer.getNodes().indexOf(node));
					System.out.println("node  value " + node.getValue());
					System.out.println("node error " + node.getError());
					System.out.println("layerWeights = ");  
					if(node.getWeights() != null) {
						int counter = 0;
						for(Double currentWeight : node.getWeights()) {
							System.out.println("layerWeight "+counter +" =|= "+currentWeight);
							counter++;
						}
					}
				}
			}
			//System.out.println("layer output = " + layer.getLayerOutput());
		}
		System.out.println("===================================================================================================================");
	}
	
	
	public void saveNNStatus(String fileName) {
		try  {
	         FileOutputStream fileOut =
	         new FileOutputStream(fileName);
	         ObjectOutputStream out = new ObjectOutputStream(fileOut);
	         out.writeObject(this);
	         out.close();
	          fileOut.close();
	      } catch(IOException i) {
	          i.printStackTrace();
	      }
	}
	
	public IqNeuralNetwork retriveNNStatusFromFile(String fileName) {
		IqNeuralNetwork nn = null;
		try {
	         FileInputStream fileIn = new FileInputStream(fileName);
	         ObjectInputStream in = new ObjectInputStream(fileIn);
	         nn = (IqNeuralNetwork) in.readObject();
	         in.close();
	         fileIn.close();
	         return nn;
	      } catch(Exception err) {
	    	  return null;
	      }
	      
	}
	
}