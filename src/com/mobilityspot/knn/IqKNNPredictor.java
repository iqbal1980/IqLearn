package com.mobilityspot.knn;

import java.util.ArrayList;
import java.util.Collections;

public class IqKNNPredictor {
	private ArrayList<IqKNNAttribute> knnAttributes;
	private IqKNNAttribute queryKnnAttribute;
 
	
	public IqKNNPredictor(ArrayList<IqKNNAttribute> inputKnnAttributes, IqKNNAttribute inpuKnntQueryAttribute) {
		this.knnAttributes = inputKnnAttributes;
		this.queryKnnAttribute = inpuKnntQueryAttribute;
	}
	
	public void predict(int kValue) {
		for(IqKNNAttribute tmp : this.knnAttributes) {
			double pow = IqKNNMath.computVectorsDistance(tmp, queryKnnAttribute);
			tmp.setDistance(pow);
			System.out.println("pow ="+ pow);
		}
		
		Collections.sort(this.knnAttributes,new IqKNNAtributeDistanceComparator());
		
		/*
		for(IqKNNAttribute tmp : this.knnAttributes) {
			double pow = IqKNNMath.computVectorsDistance(tmp, queryKnnAttribute);
			tmp.setDistance(pow);
			System.out.println("sorted ===================================="+ tmp.getDistance());
		}
		
		for(int i=0;i<this.knnAttributes.size();i++) {
			System.out.println("sorted >>>>>>>>>>>"+this.knnAttributes.get(i).getDistance());
		}
		*/
		
		
	}
}
