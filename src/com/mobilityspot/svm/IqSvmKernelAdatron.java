package com.mobilityspot.svm;


public class IqSvmKernelAdatron {
	private double[][] myInputVectors;
	private double[] myAlphas;
	private double[] myExpectedOutputs;
	private double myTetha;
	private int typeOfUsedKernel;
	
	
	static double getKernelOutput(double[] v1, double[] v2, int typeOfKernel) {
		double returnValue = 0.0;
		if(typeOfKernel == IqSvmMath.KERNEL_DOT_PRODUCT) {
			returnValue =  IqSvmMath.kernelDotProduct(v1, v2);
		}
		if(typeOfKernel == IqSvmMath.KERNEL_POLYNOMIAL) {
			returnValue =  IqSvmMath.kernelPolynomial(v1, v2);
		}
		if(typeOfKernel == IqSvmMath.KERNEL_RBF_GAUSSIAN) {
			returnValue =  IqSvmMath.kernelRBDGaussianSigmoid(v1, v2);
		}
		return returnValue;
	}
	
	
	public void svmKernelAdatronTrain(double[][] vectorsToTrain, double[] expectedOutputs, int typeOfKernel, double eta,double eps,double margin,double theta) {	
		int m = expectedOutputs.length;
		double[][] v = vectorsToTrain;
		double[] y = expectedOutputs;
		double[] alpha = new double[m];
	 
		double min = 0, max = 0;
		
		int i,j, mininit, maxinit;
		for(i=0;i<m;i++) 
		{
			alpha[i] = 1.0;
		}
		
		
		while(Math.abs(margin - 1.0) > eps) {	
			System.out.println("margin ===>"+margin);
			mininit = maxinit = 1;
			for(i=0;i<m;i++) {
			double z = 0.0;
				for(j=0;j<m;j++)  {
						z+= alpha[j]*y[j]*getKernelOutput(v[i],v[j],typeOfKernel);
				}
			double delta = eta*(1.0-y[i]*(z-theta));
			if(alpha[i]+ delta<=0.0) { 
				alpha[i] = 0.0;
			} else {
				alpha[i] += delta;
			}
			if((mininit == 1 || z<min) && y[i] >0) { min=z;mininit=0; }
			if((maxinit == 1 || z>max) && y[i] <0) {max=z;maxinit=0; }
			}
			margin=(min-max)/2.0; theta = (min+max)/2.0;
		}
		
		for(i=0;i<m;i++) {
			System.out.println("alpha["+i+"] = "+alpha[i]);
		}
		
		this.myAlphas = alpha;
		this.myExpectedOutputs = expectedOutputs;
		this.myTetha = theta;
		this.myInputVectors = vectorsToTrain;
		this.typeOfUsedKernel = typeOfKernel;
		System.out.println("theta ===>"+theta);	
	}
	
	public double classifyVectorClass(double[] vectorToClassify) {
		double finalClassificationResult = 0;
		for(int w=0; w < myAlphas.length ; w++) {
			System.out.println("v0 ===>"+myInputVectors[w][0]+" v1 ===>"+myInputVectors[w][1]);
			finalClassificationResult += myAlphas[w]*myExpectedOutputs[w]*getKernelOutput(vectorToClassify,myInputVectors[w],typeOfUsedKernel);
		}

		finalClassificationResult = finalClassificationResult - myTetha;
		System.out.println("Final Classification = "+finalClassificationResult);
		System.out.println("Final Classification with sign = "+Math.signum(finalClassificationResult));
		return Math.signum(finalClassificationResult);
	}

}
