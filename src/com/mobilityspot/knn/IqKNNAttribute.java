package com.mobilityspot.knn;

import java.util.ArrayList;
import java.util.Comparator;

public class IqKNNAttribute {
	private ArrayList<Double> attributeVectors;
	private Integer attributeCategory;
	private Double distance;
	private Integer distanceRank;
	private Boolean isQueryAttribute;
	
 
	public IqKNNAttribute(double[] inputAttributeVector,Integer inputAttributeCategory,Boolean inputIsQueryAttribute) {
		int vectorSize = inputAttributeVector.length;
		ArrayList<Double> tmp = new ArrayList<Double>(); 
		
		for(int i = 0; i < vectorSize; i++) {
			tmp.add(inputAttributeVector[i]);
		}
		
		 this.attributeVectors = tmp;
		 this.attributeCategory = inputAttributeCategory;
		 this.isQueryAttribute = inputIsQueryAttribute;
	}
	
	public ArrayList<Double> getAttributeVectors() {
		return attributeVectors;
	}

	public void setAttributeVectors(ArrayList<Double> attributeVectors) {
		this.attributeVectors = attributeVectors;
	}

	public Integer getAttributeCategory() {
		return attributeCategory;
	}
	
	public void setAttributeCategory(Integer attributeCategory) {
		this.attributeCategory = attributeCategory;
	}
	
	public Double getDistance() {
		return distance;
	}
	
	public void setDistance(Double distance) {
		this.distance = distance;
	}
	
	public Integer getDistanceRank() {
		return distanceRank;
	}
	
	public void setDistanceRank(Integer distanceRank) {
		this.distanceRank = distanceRank;
	}

 


 
	
	
}
