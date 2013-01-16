package com.mobilityspot.nn;

import java.util.ArrayList;
import java.util.Random;

public class IqNode implements java.io.Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2387967674680169002L;
	private double value;
	private ArrayList<Double> weights;
	private ArrayList<Double> gradients; // (Value of previous Node) * (Node error)
	private ArrayList<Double> lastGradients; 
	private ArrayList<Double> rpropLastDeltas;

	private double error;
	private boolean isBiasNode = false;

	public IqNode(double nodeValue) {
		this.value = nodeValue;
	}
	
	public IqNode(double nodeValue, double[] nodeWeights,double nodeError) {
		this.value = nodeValue;
		this.isBiasNode = false;

	}
	
	
	public IqNode(double nodeValue, double[] nodeWeights,double nodeError,boolean isBias) {
		this.value = nodeValue;
		this.isBiasNode = isBias;
	}

	public void initNodeWeights(int numberOfWeights ) {
		ArrayList<Double> nodeWeights = new ArrayList<Double>();
		for(int i=0 ;i< numberOfWeights ; i++) {
			//double x = (new Double(i).doubleValue())/10;
			//x = 0.50 + x;
			//nodeWeights.add(x);
			Random r=new Random();
			double myRand = r.nextDouble()*2.0 - 1.0;//(0.5 - r.nextDouble());
			nodeWeights.add(myRand);
		}
		this.weights = nodeWeights;
	}
	
	public void initNodeGradients(int numberOfGradients ) {
		ArrayList<Double> nodeGradients = new ArrayList<Double>();
		for(int i=0 ;i< numberOfGradients ; i++) {
			double x = 0;
			nodeGradients.add(x);
 
		}
		this.gradients = nodeGradients;
	}
	
	public void initNodeLastGradients(int numberOfLastGradients ) {
		ArrayList<Double> nodeLastGradients = new ArrayList<Double>();
		for(int i=0 ;i< numberOfLastGradients ; i++) {
			double x = 0;
			nodeLastGradients.add(x);
 
		}
		this.lastGradients = nodeLastGradients;
	}
	
	public void initRpropLastDeltas(int numberOfRpropLastDeltas ) {
		ArrayList<Double> nodeRpropLastDeltas = new ArrayList<Double>();
		for(int i=0 ;i< numberOfRpropLastDeltas ; i++) {
			double x = 0.1;
			nodeRpropLastDeltas.add(x);
		}
		this.rpropLastDeltas = nodeRpropLastDeltas;
	}
	
	public double getValue() {
		return value;
	}
	public void setValue(double value) {
		this.value = value;
	}

	public ArrayList<Double> getWeights() {
		return weights;
	}

	public void setWeights(ArrayList<Double> weights) {
		this.weights = weights;
	}
	
	public ArrayList<Double> getGradients() {
		return gradients;
	}

	public void setGradients(ArrayList<Double> gradients) {
		this.gradients = gradients;
	}
	
	public ArrayList<Double> getLastGradients() {
		return lastGradients;
	}

	public void setLastGradients(ArrayList<Double> lastGradients) {
		this.lastGradients = lastGradients;
	}

	public ArrayList<Double> getRpropLastDeltas() {
		return rpropLastDeltas;
	}

	public void setRpropLastDeltas(ArrayList<Double> rpropLastDeltas) {
		this.rpropLastDeltas = rpropLastDeltas;
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}

	public boolean isBiasNode() {
		return isBiasNode;
	}

	public void setBiasNode(boolean isBiasNode) {
		this.isBiasNode = isBiasNode;
	}
	
	
	
	
}
