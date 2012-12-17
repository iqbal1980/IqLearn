package com.mobilityspot.nn;

import java.util.LinkedList;

public class IqLayer implements java.io.Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6004073133128832694L;
	private LinkedList<IqNode> nodes; 
	
	
	public IqLayer(int numberOfNodes) {
		nodes = new LinkedList<IqNode>();
		
		for(int i = 0; i < numberOfNodes ; i++) {
			nodes.add(new IqNode(1, null, 0.0));
		}
	}
	
	
	public IqLayer(int numberOfNodes,boolean withBias) {
		nodes = new LinkedList<IqNode>();
		
		for(int i = 0; i < numberOfNodes ; i++) {
			nodes.add(new IqNode(1, null, 0.0));
		}		
		if(withBias == true) {
			nodes.addLast(new IqNode(1, null, 0.0, true));
		}
	}
	
	public LinkedList<IqNode> getNodes() {
		return nodes;
	}

	public void setNodes(LinkedList<IqNode> nodes) {
		this.nodes = nodes;
	}
	
 
}
