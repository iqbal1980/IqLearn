package com.mobilityspot.nn;

import java.util.LinkedList;
import java.util.Random;

public class IQLayer {
	private LinkedList<IQNode> nodes; 

	private double  getRandomWeight() {
		Random r= new Random(System.nanoTime());
		return r.nextDouble();
	}
	
	public IQLayer(int numberOfNodes) {
		nodes = new LinkedList<IQNode>();
 
		for(int i = 0; i < numberOfNodes ; i++) {
			nodes.add(new IQNode(1, getRandomWeight()));
		}
	}
	
	public double getLayerOutput() {
		double returnVal = 0;
		for(IQNode node : nodes) {
			returnVal += node.getValue() * node.getWeight();
		}
		returnVal = NNMAth.sigmoid(returnVal);
		return returnVal;
	}
	
	public LinkedList<IQNode> getNodes() {
		return nodes;
	}

	public void setNodes(LinkedList<IQNode> nodes) {
		this.nodes = nodes;
	}
	
	public void updateNodesValues(double[] values) {
		int i = 0;
		for(IQNode node : nodes) {
			node.setValue(values[i]);
			i++;
		}
	}
	
	public void updateNodesWeight() {
		
	}
}
