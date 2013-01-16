package com.mobilityspot.knn;

import java.util.ArrayList;

public class KnnTester {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//0 = bad
		//1 = good
		double[] x0 = {7,7};
		double[] x1 = {7,4};
		double[] x2 = {3,4};
		double[] x3 = {1,4};
		
		IqKNNAttribute att0 = new IqKNNAttribute(x0,0,false);
		IqKNNAttribute att1 = new IqKNNAttribute(x1,0,false);
		IqKNNAttribute att2 = new IqKNNAttribute(x2,1,false);
		IqKNNAttribute att3 = new IqKNNAttribute(x3,1,false);
		
		ArrayList<IqKNNAttribute> test = new ArrayList<IqKNNAttribute>();
		test.add(att0);
		test.add(att1);
		test.add(att2);
		test.add(att3);
		
		double[] xQuery = {3,7};
		IqKNNAttribute queryAtt = new IqKNNAttribute(xQuery,null,true);
		
		IqKNNPredictor predictor = new IqKNNPredictor(test, queryAtt);
		predictor.predict(3);//k = 3 nearest neighbours

	}

}
