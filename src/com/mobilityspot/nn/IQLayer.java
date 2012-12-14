package com.mobilityspot.nn;

import java.util.LinkedList;

public class IQLayer implements java.io.Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6004073133128832694L;
	private LinkedList<IQNode> nodes; 
	
	
	public IQLayer(int numberOfNodes) {
		nodes = new LinkedList<IQNode>();
		
		for(int i = 0; i < numberOfNodes ; i++) {
			nodes.add(new IQNode(1, null, 0.0));
		}
	}
	
	
	public IQLayer(int numberOfNodes,boolean withBias) {
		nodes = new LinkedList<IQNode>();
		
		for(int i = 0; i < numberOfNodes ; i++) {
			nodes.add(new IQNode(1, null, 0.0));
		}		
		if(withBias == true) {
			nodes.addLast(new IQNode(1, null, 0.0, true));
		}
	}
	
	public LinkedList<IQNode> getNodes() {
		return nodes;
	}

	public void setNodes(LinkedList<IQNode> nodes) {
		this.nodes = nodes;
	}
	
 
}
