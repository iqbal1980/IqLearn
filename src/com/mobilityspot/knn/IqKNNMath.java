package com.mobilityspot.knn;

public class IqKNNMath {

	static double computVectorsDistance(IqKNNAttribute v1, IqKNNAttribute v2) {
		double distance = 0;
		for(int i =0; i< v1.getAttributeVectors().size(); i++) {
			distance +=  Math.pow( (v1.getAttributeVectors().get(i).doubleValue() - v2.getAttributeVectors().get(i).doubleValue()), 2);
 
		}
		return distance;
	}
}
