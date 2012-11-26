package com.mobilityspot.nn;

import java.util.LinkedList;

public class IQLayer {
	private LinkedList<IQNode> nodes; 
	
	
	public IQLayer(int numberOfNodes) {
		nodes = new LinkedList<IQNode>();
		
		for(int i = 0; i < numberOfNodes ; i++) {
			nodes.add(new IQNode(1, null, 10));
		}
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
	
	public void updateNodesValues(double sameValue) {
		int i = 0;
		for(IQNode node : nodes) {
			node.setValue(sameValue);
			i++;
		}
	}
}
