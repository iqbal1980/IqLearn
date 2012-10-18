package com.mobilityspot.nn;

public class IQNode {
	private double value;
	private double weight;
/*
	public IQNode(1) {
		this.value = 1;
		this.weight = ;
	}*/
	
	public IQNode(double nodeValue, double nodeWeight) {
		this.value = nodeValue;
		this.weight = nodeWeight;
	}
	
	public double getValue() {
		return value;
	}
	public void setValue(double value) {
		this.value = value;
	}
	public double getWeight() {
		return weight;
	}
	public void setWeight(double weight) {
		this.weight = weight;
	}

}
