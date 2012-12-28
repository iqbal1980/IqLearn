package com.mobilityspot.svm;

public class TestSvm {

	public TestSvm() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		IqSvmKernelAdatron svm = new IqSvmKernelAdatron();
		double[][] v = {{-1,-1}, {-1,+1}, {+1,-1}, {+1,+1}};
		double[] y = {+1, -1, -1, +1};
		double[] alpha = new double[y.length];
		double eta = 0.01, eps = 0.00001, margin = 0.0, theta = 0.0;
		double[] toClassifyVector = {-1.8, -0.95};
 
		svm.svmKernelAdatronTrain( v, y, IqSvmMath.KERNEL_POLYNOMIAL,0.01,0.00001,0.0,theta);
		svm.classifyVectorClass(toClassifyVector);

	}

}
