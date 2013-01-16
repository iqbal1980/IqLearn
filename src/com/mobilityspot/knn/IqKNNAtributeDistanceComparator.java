package com.mobilityspot.knn;

import java.util.Comparator;

public class IqKNNAtributeDistanceComparator implements Comparator<IqKNNAttribute> {
	@Override
	public int compare(IqKNNAttribute o1, IqKNNAttribute o2) {
		 return (o1.getDistance()<o2.getDistance() ? -1 : (o1.getDistance()==o2.getDistance() ? 0 : 1));
	}
}
