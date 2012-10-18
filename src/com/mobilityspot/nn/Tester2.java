package com.mobilityspot.nn;

public class Tester2 {

	public Tester2() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		double[][] inputs = {{1,2},{2,1},{3,2},{3,7}};
		System.out.println(inputs.length);
		for(int i=0;i < inputs.length ; i++) {
			for(int j=0;j < inputs[i].length ; j++) {
				System.out.println(inputs[i][j]);
			}
		}
		

	}

}
