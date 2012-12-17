package com.mobilityspot.svm;

public class IqSvmMath {
	
	static int KERNEL_DOT_PRODUCT 	=  0;
	static int KERNEL_POLYNOMIAL  	=  1;
	static int KERNEL_RBF_GAUSSIAN  =  2;

	static double kernelDotProduct(double[] v1, double[] v2) {
		double k = 0;
		for(int counter = 0;counter < v1.length; counter++) {
			k += v1[counter]*v2[counter];
		}
		return k;
	}
	
	static double kernelPolynomial(double[] v1, double[] v2) {
		double k = 1.0 + IqSvmMath.kernelDotProduct(v1, v2);
		
		return k*k;
	}
	
	static double kernelRBDGaussianSigmoid(double[] v1, double[] v2) {
		double  k=0;
	      for(int counter=0;counter<v1.length;counter++) { 
	    	  k += (v1[counter] - v2[counter])*(v1[counter] - v2[counter]); 
	      } 
	      return Math.exp(-0.5*(k));
	      //return Math.exp(-k/(2.0*0.5*0.5));//sigma = 0.5??
	}
	
	

}
