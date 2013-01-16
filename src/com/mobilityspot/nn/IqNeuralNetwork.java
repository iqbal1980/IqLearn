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
	private static final String IqNeuralNetworkMath2 = null;
	private LinkedList<IqLayer> layers  = new LinkedList<IqLayer>();
	private double[][] inputs;
	private double[][] expectedOutputs;
	private int nnTrainingEpoches;
	private boolean isDebugOn = false;
	private int trainingMethod = 0;
	
	private int trainingBatchIndex = 0;
	
	//private double lastGradient1 = 0;
	//private double lastGradient2 = 0;
	//private double lastDelta1 = 0.1;
	//private double lastDelta2 = 0.1;

 
	
 
	
	
	public IqNeuralNetwork(double[][] myInputs, double[][] myExpectedOutputs,  int[] dimensionsOfHiddenLayers , int trainingEpoches,int chosenTrainingMethod, boolean debugOn) throws Exception {
		this.inputs = myInputs;
		this.expectedOutputs = myExpectedOutputs;
		this.nnTrainingEpoches = trainingEpoches;
		this.isDebugOn = debugOn;
		this.trainingMethod = chosenTrainingMethod;
		
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
		
		IqLayer outputLayer = new IqLayer(numberOfOutpus,false);
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
					myNode.initNodeGradients(previousLayerSize);
					myNode.initNodeLastGradients(previousLayerSize);
					myNode.initRpropLastDeltas(previousLayerSize);
				}
			}
		}
		
		printNNStructure("initialization");
		
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
		for(int i = 0; i < this.nnTrainingEpoches  ; i++) { //training iterations
			this.trainingBatchIndex = 0;
			for(int j = 0; j<  inputs.length    ;j++) {
				this.trainingBatchIndex = j;
				netWorkOutPut = getNetworkOutput(inputs[j]);
				if(this.isDebugOn == true) {
					printNNStructure("feedForward");
				}
				double errorRateBeforeAllowingBackPropagation = 0;
				for(int k=0;k<netWorkOutPut.length - 1;k++) {	
				    errorRateBeforeAllowingBackPropagation += netWorkOutPut[k]/this.expectedOutputs[j][k];
				}
				
				if(this.trainingMethod == IqNeuralNetworkMath.IRPROPMIN) {
					backPropagateErrorWithRPROP(netWorkOutPut,this.expectedOutputs[j]);
				} else {
					//if(errorRateBeforeAllowingBackPropagation > 0.2) {
						//backPropagateError(netWorkOutPut,this.expectedOutputs[j]);
						backPropagateErrorBatch(netWorkOutPut,this.expectedOutputs[j]);
					//}	
				}
				

				
				if(this.isDebugOn == true) {
				  printNNStructure("backPropagation");
				}
			}
		}
	}
	
	public double[]  getNetworkOutput(double[] inputsValues) {
		for(int i=0 ;i < layers.getFirst().getNodes().size()  - 1 ; i++) {//-1 accounts for disregarding bias node
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
		return returnValue;

	}
	
	
	public void backPropagateError(double[] nnOutput, double[] nnExpectedOutputs) {
		for(int i=0 ;i < layers.getLast().getNodes().size()/* - 1*/; i++) {//-1 accounts for disregarding bias node
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
	
	
	public void backPropagateErrorBatch(double[] nnOutput, double[] nnExpectedOutputs) {
		for(int i=0 ;i < layers.getLast().getNodes().size()/* - 1*/; i++) {//-1 accounts for disregarding bias node
			double currentErrorDelta = 0;
			currentErrorDelta = layers.getLast().getNodes().get(i).getValue();
			currentErrorDelta = currentErrorDelta*(1 - currentErrorDelta)*(nnExpectedOutputs[i] - currentErrorDelta);
 
			
			
			layers.getLast().getNodes().get(i).setError(currentErrorDelta);
			
			ArrayList<Double> updatedWeights = new ArrayList<Double>(layers.getLast().getNodes().get(i).getWeights().size());
			ArrayList<Double> updatedGradients = new ArrayList<Double>(layers.getLast().getNodes().get(i).getGradients().size());
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			for(int j = 0 ; j < layers.getLast().getNodes().get(i).getWeights().size(); j++) {
				double myWeight = 0;
				double myGradient = layers.getLast().getNodes().get(i).getGradients().get(j).doubleValue();;
				int layersSize = layers.size() - 1;
				
				myWeight = layers.getLast().getNodes().get(i).getWeights().get(j).doubleValue();
				myGradient += (layers.getLast().getNodes().get(i).getError()) * layers.get(layersSize - 1).getNodes().get(j).getValue();
				
				//System.out.println("this.trainingBatchIndex =====> "+this.trainingBatchIndex+" || gradient =====> "+myGradient);
				if(this.trainingBatchIndex == 3) {
					myWeight = IqNeuralNetworkMath.LEARNING_MOMENTUM*myWeight + IqNeuralNetworkMath.LEARNING_RATE * myGradient;
				 }
				updatedWeights.add(myWeight);
				updatedGradients.add(myGradient);
			}
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			layers.getLast().getNodes().get(i).getWeights().clear();
			layers.getLast().getNodes().get(i).setWeights(updatedWeights);	
			
			layers.getLast().getNodes().get(i).getGradients().clear();
			layers.getLast().getNodes().get(i).setGradients(updatedGradients);	
		}
 
		
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
					 ArrayList<Double> newGradients = new ArrayList<Double>();
					 
					 newWeights.clear();
					 newGradients.clear();
					 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					 for(int n = 0; n < layers.get(k).getNodes().get(l).getWeights().size(); n++) {
						 double newNodeWeight = 0;
						 double newNodeGradient = layers.get(k).getNodes().get(l).getGradients().get(n).doubleValue();
						 newNodeWeight = layers.get(k).getNodes().get(l).getWeights().get(n).doubleValue();
						 newNodeGradient += layers.get(k).getNodes().get(l).getError()*layers.get(k-1).getNodes().get(n).getValue();
						 if(this.trainingBatchIndex == 3) {
							 newNodeWeight = IqNeuralNetworkMath.LEARNING_MOMENTUM*newNodeWeight + IqNeuralNetworkMath.LEARNING_RATE*newNodeGradient;
						 }	 
						 newWeights.add(newNodeWeight);
						 newGradients.add(newNodeGradient);
					 }
					 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					 layers.get(k).getNodes().get(l).getWeights().clear();
					 layers.get(k).getNodes().get(l).setWeights(newWeights);
					 
					 layers.get(k).getNodes().get(l).getGradients().clear();
					 layers.get(k).getNodes().get(l).setGradients(newGradients);
				 }
			}
		}
	
	}	
	

	public void backPropagateErrorWithRPROP(double[] nnOutput, double[] nnExpectedOutputs) {

		
		for(int i=0 ;i < layers.getLast().getNodes().size() /* - 1*/; i++) {//-1 accounts for disregarding bias node
			double currentErrorDelta = 0;
			currentErrorDelta = layers.getLast().getNodes().get(i).getValue();
			
			currentErrorDelta = currentErrorDelta*(1 - currentErrorDelta)*(nnExpectedOutputs[i] - currentErrorDelta);
			layers.getLast().getNodes().get(i).setError(currentErrorDelta);

			
			ArrayList<Double> updatedWeights = new ArrayList<Double>(layers.getLast().getNodes().get(i).getWeights().size());
			ArrayList<Double> updateGradients = new ArrayList<Double>(layers.getLast().getNodes().get(i).getGradients().size());
			ArrayList<Double> updateLastGradients = new ArrayList<Double>(layers.getLast().getNodes().get(i).getLastGradients().size());
			ArrayList<Double> updateRpropLastDeltas = new ArrayList<Double>(layers.getLast().getNodes().get(i).getRpropLastDeltas().size());

			//////////////////////////
			//double lastGradient1 = 0;
			//double lastDelta1 = 0.1;
			//////////////////////////

			for(int j = 0 ; j < layers.getLast().getNodes().get(i).getWeights().size(); j++) {
				
				int layersSize = layers.size() - 1;
				

				double myWeight = layers.getLast().getNodes().get(i).getWeights().get(j).doubleValue();
				double myGradient = (layers.getLast().getNodes().get(i).getError()) * layers.get(layersSize - 1).getNodes().get(j).getValue();
				double myLastGradient = layers.getLast().getNodes().get(i).getLastGradients().get(j).doubleValue();
				double myPropLastDelta = layers.getLast().getNodes().get(i).getRpropLastDeltas().get(j).doubleValue();

				System.out.println("current node error1 == "+layers.getLast().getNodes().get(i).getError());
				System.out.println("previous node/weight value1 == "+layers.get(layersSize - 1).getNodes().get(j).getValue());
				System.out.println("layer last || node = " +i+	" || weight===> "+j+" || myGradient == "+myGradient+" || myLastGradient == "+myLastGradient+" lastdelta == "+myPropLastDelta);
				
				////////////////////////////////////////////////////////////////////////////
				final int change1 = IqNeuralNetworkMath.sign(myGradient * myLastGradient);
				System.out.println("change1 ======>"+change1);
                double weightChange1 = 0;
                double delta1;

                if (change1 > 0) {
                        delta1 = myPropLastDelta * 1.2;
                        delta1 = Math.min(delta1, 50); 
                        
                        System.out.println("change1 >0 ===> MAX delta1 = 1.2*myPropLastDelta == "+myPropLastDelta);
                } else  {
                        delta1 = myPropLastDelta * 0.5;
                        delta1 = Math.max(delta1, 0.000001);
                        myLastGradient = 0;
                        
                        System.out.println("change1 else ===> MIN delta1 = 0.5*myPropLastDelta == "+myPropLastDelta);
                } 
                
                System.out.println("delta1=====>"+delta1);
                myLastGradient = myGradient;
                weightChange1 =  IqNeuralNetworkMath.sign(myGradient) * delta1;
                myWeight += weightChange1;
                myPropLastDelta = delta1;
                
				////////////////////////////////////////////////////////////////////////////	
                
                System.out.println("delta1 after=====>"+delta1);
				
				updatedWeights.add(myWeight);
				updateGradients.add(myGradient);
				updateLastGradients.add(myLastGradient);
				updateRpropLastDeltas.add(myPropLastDelta);
				
			}

			layers.getLast().getNodes().get(i).getWeights().clear();
			layers.getLast().getNodes().get(i).setWeights(updatedWeights);	

			layers.getLast().getNodes().get(i).getGradients().clear();
			layers.getLast().getNodes().get(i).setGradients(updateGradients);
			
			layers.getLast().getNodes().get(i).getLastGradients().clear();
			layers.getLast().getNodes().get(i).setLastGradients(updateLastGradients);
			
			layers.getLast().getNodes().get(i).getRpropLastDeltas().clear();
			layers.getLast().getNodes().get(i).setRpropLastDeltas(updateRpropLastDeltas);
		}
 

		for(int k = (layers.size() - 2 ); k >=0 ; k--) {//looping trough middle hidden layers
			if(layers.listIterator(k).hasPrevious() == true) {
				 for(int l = 0; l < layers.get(k).getNodes().size() -1 ; l++) {//looping trough nodes of layer //disregarding bias node
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
					 ArrayList<Double> newGradients = new ArrayList<Double>();
					 ArrayList<Double> newLastGradients = new ArrayList<Double>();
					 ArrayList<Double> newRpropLastDeltas = new ArrayList<Double>();

					 newWeights.clear();
					 newGradients.clear();
					 newLastGradients.clear();
					 newRpropLastDeltas.clear();
					//////////////////////////
					//double lastGradient2 = 0;
					//double lastDelta2 = 0.1;
					//////////////////////////


					 for(int n = 0; n < layers.get(k).getNodes().get(l).getWeights().size(); n++) {
						 double newNodeWeight = layers.get(k).getNodes().get(l).getWeights().get(n).doubleValue();
						 double newNodeGradient = layers.get(k).getNodes().get(l).getError()*layers.get(k-1).getNodes().get(n).getValue();
						 double newNodeLastGradient = layers.get(k).getNodes().get(l).getLastGradients().get(n).doubleValue();
						 double newRpropLastDelta = layers.get(k).getNodes().get(l).getRpropLastDeltas().get(n).doubleValue();
						 
						 
						 System.out.println(">>>>>>>>>>>>>>>>>current node error2 == "+layers.get(k).getNodes().get(l).getError());
						 System.out.println(">>>>>>>>>>>>>>>>>previous node/weight value2 == "+layers.get(k-1).getNodes().get(n).getValue());
						 System.out.println(">>>>>>>>>>>>>>>>>layer = " + k +" || node = " +l+	" || weight===> "+n+" || newNodeGradient == "+newNodeGradient+" || newNodeLastGradient == "+newNodeLastGradient+" lastdelta == "+newRpropLastDelta);
						 
						 
					 		////////////////////////////////////////////////////////////////////////////
						    final int change2 = IqNeuralNetworkMath.sign(newNodeGradient * newNodeLastGradient);
						    System.out.println(">>>>>>>>>>>>>>>>>delta2======>"+change2);
			                double weightChange2 = 0;
			                double delta2;


			                if (change2 > 0) {
			                	 	delta2 = newRpropLastDelta * 1.2;
			                        delta2 = Math.min(delta2, 50);   
			                        
			                        System.out.println(">>>>>>>>>>>>>>>>>change2 >0 ===> MAX delta2 = 1.2*newRpropLastDelta == "+newRpropLastDelta);
			                } else  {
			                        delta2 = newRpropLastDelta * 0.5;
			                        delta2 = Math.max(delta2, 0.000001);
			                        newNodeLastGradient =0;
			                        
			                        System.out.println(">>>>>>>>>>>>>>>>>change2 else ===> MIN delta2 = 0.5*newRpropLastDelta == "+newRpropLastDelta);
			                }

			                newNodeLastGradient = newNodeGradient;
			                weightChange2 = IqNeuralNetworkMath.sign(newNodeGradient) * delta2;
			                newNodeWeight += weightChange2;
			                newRpropLastDelta = delta2;
							////////////////////////////////////////////////////////////////////////////						 
						
			                System.out.println(">>>>>>>>>>>>>>>>>delta2 after =====>"+delta2);
						 
						 newWeights.add(newNodeWeight);
						 newGradients.add(newNodeGradient);
						 newLastGradients.add(newNodeLastGradient);
						 newRpropLastDeltas.add(newRpropLastDelta); 
					 }
					 layers.get(k).getNodes().get(l).getWeights().clear();
					 layers.get(k).getNodes().get(l).setWeights(newWeights);

					 layers.get(k).getNodes().get(l).getGradients().clear();
					 layers.get(k).getNodes().get(l).setGradients(newGradients);
					 
					 layers.get(k).getNodes().get(l).getLastGradients().clear();
					 layers.get(k).getNodes().get(l).setLastGradients(newLastGradients);
					 
					 layers.get(k).getNodes().get(l).getRpropLastDeltas().clear();
					 layers.get(k).getNodes().get(l).setRpropLastDeltas(newRpropLastDeltas);
				 }
			}
		}
	}

	
 
	
	
	public void printNNStructure(String title) {
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println("=================================== START: <<< "+title.toUpperCase()+" >>>=======================================");
		System.out.println("===================================================================================================================");
		for(IqLayer layer : layers) {
			System.out.println("layer **************"+ layers.indexOf(layer));
			for(IqNode node : layer.getNodes()) {
				if(   layers.indexOf(layer) == 0 || layers.indexOf(layer) == (layers.size() - 1)  ) {
					System.out.println("node number " + layer.getNodes().indexOf(node));
					System.out.println("	node  value " + node.getValue());
					System.out.println("		node error " + node.getError());
					System.out.println(" 			layerWeights = ");  
					if(node.getWeights() != null) {
						int counter = 0;
						for(Double currentWeight : node.getWeights()) {
							System.out.println("				layerWeight "+counter +" =|= "+currentWeight);
							counter++;
						}
					}
					
					System.out.println("			layerGradients = ");  
					if(node.getWeights() != null) {
						int counter = 0;
						for(Double currentGradient : node.getGradients()) {
							System.out.println("				layerGradient "+counter +" =|= "+currentGradient);
							counter++;
						}
					}
					/*
					System.out.println("			layerLastGradients = ");  
					if(node.getWeights() != null) {
						int counter = 0;
						for(Double currentlayerLastGradient : node.getLastGradients()) {
							System.out.println("				layerLastGradient "+counter +" =|= "+currentlayerLastGradient);
							counter++;
						}
					}
					
					System.out.println("			layerRpropLastDeltas = ");  
					if(node.getWeights() != null) {
						int counter = 0;
						for(Double currentRpropLastDelta : node.getRpropLastDeltas()) {
							System.out.println("				rpropLastDeltas "+counter +" =|= "+currentRpropLastDelta);
							counter++;
						}
					}
					*/
				}
			}
			//System.out.println("layer output = " + layer.getLayerOutput());
		}
		System.out.println("===================================================================================================================");
		System.out.println("=================================== END: <<< "+title.toUpperCase()+" >>>=======================================");
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
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

