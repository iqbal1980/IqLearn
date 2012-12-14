package com.mobilityspot.svm;

public class TestSVM {

 
			
	public TestSVM() {
		 
	}

	static double getKernelOutput(double[] vi, double[] vj) {
		double k1 = 1.0 + vi[0]*vj[0] + vi[1]*vj[1];
		return k1*k1;
	}
 
	public static void main(String[] args) {
	 
		int m = 4;
		double[][] v = {{-1,-1}, {-1,+1}, {+1,-1}, {+1,+1}};
		double[] y = {+1, -1, -1, +1};
		double[] alpha = new double[m];
		double eta = 0.01, eps = 0.00001, margin = 0.0, theta = 0.0;
		double min = 0, max = 0;
		
		int i,j, mininit, maxinit;
		for(i=0;i<m;i++) 
		{
			alpha[i] = 1.0;
		}
		
		
		while(Math.abs(margin - 1.0) > eps) 
		{	
			System.out.println("margin ===>"+margin);
			mininit = maxinit = 1;
			for(i=0;i<m;i++)
			{
			double z = 0.0;
				for(j=0;j<m;j++) 
					{
						z+= alpha[j]*y[j]*getKernelOutput(v[i],v[j]);
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
		System.out.println("theta ===>"+theta);
		
	 
		double finalClassificationResult = 0;
		//classifications:
		for(int w=0; w < alpha.length; w++) {
			System.out.println("v0 ===>"+v[w][0]+" v1 ===>"+v[w][1]);
			double[] toClassifyVector = {-0.95, 1.0};
			finalClassificationResult += alpha[w]*y[w]*getKernelOutput(toClassifyVector,v[w]);
		}
		
		finalClassificationResult = finalClassificationResult - theta;
		
		System.out.println("Final Classification = "+finalClassificationResult);
		
	}

}
