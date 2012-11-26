package com.mobilityspot.nn;

import java.util.ArrayList;

public class IQNode {
	private double value;
	private ArrayList<Double> weights;
	private double error;

	public IQNode(double nodeValue) {
		this.value = nodeValue;
	}
	
	public IQNode(double nodeValue, double[] nodeWeights,double nodeError) {
		this.value = nodeValue;

	}
	
	
	public void initNodeWeights(int numberOfWeights ) {
		ArrayList<Double> nodeWeights = new ArrayList<Double>();
		for(int i=0 ;i< numberOfWeights ; i++) {
			nodeWeights.add(0.2);
		}
		this.weights = nodeWeights;
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
	
	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}
	
	
}
