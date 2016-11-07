/**************************
 Quarter   : Spring 2015 CSCI 571
 Created   : 5-23-15
 Author    : Aaron Tuor
 Program   : Prog3.java- implementation of basic neural network
 ---------------------------
 ***************************/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.lang.Math.*;
public class Prog3 {
	//stuff we need
	private static int ntrain;//number of training points
	private static int ndev;//number of dev points
	private static int D;//dimension of data
	private static int C;//dimension of target vectors
    private static int iterCounter = 0;//# of epochs trained on  
	private static int MAX_BAD_COUNT;//max number of non-improving dev after gradient update
	private static int badcount = 0;//current number of non-improving dev after gradient update
	private static double bestDevLoss = Double.POSITIVE_INFINITY;//to determine bad count
	private static int hiddenSize;//number of nodes in hidden layer
    private static double[][] devY;//target vectors of dev set
	private static double[][] trainY;//target vectors of train set
	private static double[] a;// W^T[1 x]^T   
	private static double[] b;// U^T[1 h]^T
	private static double[] h;// sigmoid(a)
	private static double[] hplus1;// [1 h]
	private static double[][] W; //linear transformation
	private static double[][] U; //linear transformation
	private static double[] deltab; // gradient wrt b
	private static double[] deltaa;	// gradient wrt a
	private static int MAX_ITERATIONS; // maximum number of epochs to train on
	private static String printstyle;
//---------------------------------------------------------------------------------------------	
	public static void main(String[] args) throws FileNotFoundException {
		//stuff we need		
		HashMap<String,String> argstruct = ArgParse.validateArgs(args);
		printstyle = argstruct.get("print");		
		MAX_ITERATIONS = Integer.parseInt(argstruct.get("epochs"));	
		MAX_BAD_COUNT = Integer.parseInt(argstruct.get("badcount"));	
		ntrain = Integer.parseInt(argstruct.get("N_TRAIN"));
		ndev = Integer.parseInt(argstruct.get("N_DEV"));
		D = Integer.parseInt(argstruct.get("D"));
		C = Integer.parseInt(argstruct.get("C"));		
		String trainFeatureFileName = argstruct.get("TRAIN_X_FN");
		String trainTargetFileName = argstruct.get("TRAIN_T_FN");
		String devFeatureFileName = argstruct.get("DEV_X_FN");
		String devTargetFileName = argstruct.get("DEV_T_FN");
		String isStandardized = argstruct.get("standard");
		int batchSize = Integer.parseInt(argstruct.get("batch"));
		String mo = argstruct.get("mo");
		String func2 = argstruct.get("activation");
		String func1 = argstruct.get("non-linearity"); 
		double stepsize = Double.parseDouble(argstruct.get("stepsize"));
		hiddenSize = Integer.parseInt(argstruct.get("L"));	
		boolean printepoch = false;
		
		//because lab specified this		
		if (batchSize == 0) {
			batchSize = ntrain;
		}
		//stuff we don't do		
		if (func1.equals("tanh")) {
			quitProgram("Prog3 does not support hyperbolic tangent non-linearity");
		}
		if (! mo.equals("false")) {
			quitProgram("Prog3 does not support momentum parameters");
		}
		
		/* Read Training and Dev Data */
		double[][] trainP = readDataFile(trainFeatureFileName, "x", ntrain, D);
		double[][] devP = readDataFile(devFeatureFileName, "x", ndev, D);
		double[][] trainT = readDataFile(trainTargetFileName, "t", ntrain, C);
		double[][] devT = readDataFile(devTargetFileName, "t", ndev, C);		
		
		//initialize W
		double upperbound = 1.0/Math.sqrt((double)(D + 1));
		double lowerbound = -1.0*upperbound;	
		W = randomMatrix(D+1, hiddenSize, upperbound, lowerbound);
	
		//initialize U
		upperbound = 1.0/Math.sqrt((double)(hiddenSize + 1));
		lowerbound = -1.0*upperbound;
		U = randomMatrix(hiddenSize + 1, C, upperbound, lowerbound);
		int epochCounter = 0;
		int count = 0;
		while (epochCounter < MAX_ITERATIONS && badcount < MAX_BAD_COUNT) {
			//initialize gradient matrices
			double[][] gU = new double[U.length][U[0].length];
			double[][] gW = new double[W.length][W[0].length];						
			
			for(int i = 0; i < batchSize; i++) {
				double[] current_x = trainP[count];								
				double[] y = feedforward(current_x, func2);
				double[] x_target = trainT[count];							
				gU = matAdd(backpropagate1(x_target, y), gU);
				gW = matAdd(backpropagate2(current_x), gW);
				count++;
				if (count == ntrain -1) {
					epochCounter++;
					count = 0;
					if (printstyle.equals("quiet")) {
						printepoch = true;
					}
				}			
			}
			iterCounter++;			
			gU = scalarMultiply(gU , 1.0/(double)batchSize);
			gW = scalarMultiply(gW , 1.0/(double)batchSize);
			U = matMinus(U, scalarMultiply(gU, stepsize));
			W = matMinus(W, scalarMultiply(gW, stepsize));			
			//check loss			
			devY = feedforwardMatrix(devP, func2);			 
			double devAverageLoss = getAverageLoss(devY, devT, func2);
			if(! printstyle.equals("quiet")) {	
				trainY = feedforwardMatrix(trainP, func2);		
				printAssignments(printstyle, devAverageLoss);
			} else if (printepoch) {
				printAssignments(printstyle, devAverageLoss);
				printepoch = false;
			}
			if (devAverageLoss <= bestDevLoss) {
				badcount = 0;
				bestDevLoss = devAverageLoss;
			} else {
				badcount++;
			}
		}
		devY = feedforwardMatrix(devP, func2);
		trainY = feedforwardMatrix(trainP, func2);
		double devLoss = getAverageLoss(devY, devT, func2);
		double trainLoss = getAverageLoss(trainY, trainT, func2);		
		System.err.print("Final ");		
		System.err.print(String.format("trainObj=%.5f ", trainLoss));
		System.err.println(String.format("devObj=%.5f", devLoss));
	//----------------------------------------------------------------------------------------------------
	}//end main

//neural network training steps

	public static double[] feedforward(double[] point, String func2) {		
		double[][] test = transpose(W);
		a = matVecMultiply(transpose(W), point);
		h = vecSigmoid(a);	
		hplus1 = pad1(h);
		b = matVecMultiply(transpose(U), hplus1); 
		return activation(func2, b);
	}
	
	//runs whole matrix through neural net and returns a predictions matrix
	public static double[][] feedforwardMatrix(double[][] points, String func2) {
		double[][] workingMatrix = matMultiply(points, W);
		workingMatrix = matSigmoid(workingMatrix);
		double[][] workingMatrix2 = pad1Matrix(workingMatrix);
		double[][] guesses = matMultiply(workingMatrix2, U);
		guesses = activation(func2, guesses);
		return guesses;
	}
	
	//finds grad(U)	
	public static double[][] backpropagate1(double[] target, double[] guess) {		
		deltab = vecSubtract(target, guess);
		double[][] dbMat = new double[1][deltab.length];
		dbMat[0] = deltab;
		double[][] hplus1MatTranspose = new double[1][hplus1.length];
		hplus1MatTranspose[0] = hplus1;
		double[][] negHplus1Mat = scalarMultiply(transpose(hplus1MatTranspose), -1);
		double [][] gradientU = matMultiply(negHplus1Mat, dbMat);
		return gradientU;  		
	}

	//finds grad(W)
	public static double[][] backpropagate2(double[] xplus1) {
		double[] sigmoida = vecSigmoid(a);
		double[] ones = new double[a.length];
		Arrays.fill(ones, 1);		
		double[] sigmoidOneMinusa = vecSigmoid(vecSubtract(ones, a));		
		double[] GRADfWRTa = elementVecMultiply(sigmoidOneMinusa, sigmoida);		
		deltaa = elementVecMultiply(removeFirstIndex(matVecMultiply(U, deltab)), GRADfWRTa);
		double[][] daMat = new double[1][deltaa.length];
		daMat[0] = deltaa;
		double[][] xplus1MatTranspose = new double[1][xplus1.length];
		xplus1MatTranspose[0] = xplus1;
		double[][] negxplus1Mat = scalarMultiply(transpose(xplus1MatTranspose), -1);
		double[][] gradientW = matMultiply(negxplus1Mat, daMat);
		return gradientW;
	}

	//computes Average loss
	public static double getAverageLoss(double[][] fedForwardSet, double[][] targetset, String func) { 		
		double totalLoss = 0.0;		
				
		for (int i = 0; i < targetset.length; i++) {
			totalLoss += loss(func, fedForwardSet[i], targetset[i]);
		}		
		return totalLoss /= (double)targetset.length;
	}
	
	//variable activation function for vectors
	private static double[] activation(String function, double[] b) {
		if (function.equals("softmax")){
			return vecSoftmax(b);
		}
		if (function.equals("sigmoid")){
			return vecSigmoid(b);
		}
		if (function.equals("identity")){
			return b;
		}
		return null;
	} 

	//variable activation function for matrices
	private static double[][] activation(String function, double[][] B) {
		if (function.equals("softmax")){
			return matrixSoftmax(B);
		}
		if (function.equals("sigmoid")){
			return matSigmoid(B);
		}
		if (function.equals("identity")){
			return B;
		}
		return null;
	} 

	//variable loss function for a single data point
	private static double loss(String function, double[] y, double[] t) {
		double loss = 0.0;		
		if (function.equals("softmax")){
			for (int i = 0; i < C; i++) {
				loss += t[i]*Math.log(y[i]);
			}
			loss *= -1;
		}
		if (function.equals("sigmoid")){
			for (int i = 0; i < C; i++) {
				loss += t[i] * Math.log(y[i]) + (1 - t[i])* Math.log(1 - y[i]);
			}
			loss *= -1;
		}
		if (function.equals("identity")){
			for (int i = 0; i < C; i++) {
				loss += Math.pow((y[i] - t[i]), 2.0);
			}
		}
		return loss;
	} 

//matrix, vector, and functional utilities for neural network computations.--------------------------------------------
    
	//randomMatrix: creates random matrix of specified dimensions and random range	
	public static double[][] randomMatrix(int rowsize, int colsize, double upperbound, double lowerbound) {
		Random rand = new Random();		
		double[][] randMatrix = new double[rowsize][colsize]; 		
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				randMatrix[i][j] = rand.nextDouble()*(upperbound - lowerbound) + lowerbound;
			}
		}
		return randMatrix;
	}    	

	//dot product
	private static double xDoty(double[] x, double[] y) {
        double z = 0;
        for (int i = 0; i < y.length; i++) {
            z += x[i] * y[i];
        }
        return z;
    }
   	
	//transpose
	private static double[][] transpose(double[][] X) {
		int rowsize = X.length;
		int colsize = X[0].length;
		double[][] Y = new double[colsize][rowsize];
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				Y[j][i] = X[i][j];
			}
		}
		return Y; 
	}
	
	//matrix vector multiplication
	private static double[] matVecMultiply(double[][] X, double[] y) {
		int rowsize = X.length;	
		int colsize = X[0].length;	
		double[] z = new double[rowsize];		
		for (int i = 0; i < rowsize; i++) {
			z[i] = xDoty(X[i], y);
		} 
		return z;
	}

	//matrix multiplication
	private static double[][] matMultiply(double[][] X, double[][] Y) {
		int rowsize = X.length;	
		int colsize = Y[0].length;
		double [][] Ytranspose = transpose(Y);	
		double[][] Z = new double[rowsize][colsize];		
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				Z[i][j] = xDoty(Ytranspose[j], X[i]);
			}
		} 
		return Z;
	}
	
	//vector subtraction
	private static double[] vecSubtract(double[] x, double[] y) {
		double[] z = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			z[i] = x[i] - y[i];
		}
		return z;
	}

	//vector addition
	private static double[] vecAdd(double[] x, double[] y) {
		double[] z = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			z[i] = x[i] + y[i];
		}
		return z;
	}

	//elementwise vector multiplication
	private static double[] elementVecMultiply(double[] x, double[] y) {
		double[] z = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			z[i] = x[i] * y[i];
		}
		return z;
	}

	//matrix addition
	private static double[][] matAdd(double[][] X, double[][] Y) {
		int rowsize = X.length;	
		int colsize = Y[0].length;	
		double[][] Z = new double[rowsize][colsize];		
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				Z[i][j] = X[i][j] + Y[i][j];
			}
		} 
		return Z;
	}

	//matrix subtraction
	private static double[][] matMinus(double[][] X, double[][] Y) {
		int rowsize = X.length;	
		int colsize = Y[0].length;	
		double[][] Z = new double[rowsize][colsize];		
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				Z[i][j] = X[i][j] - Y[i][j];
			}
		} 
		return Z;
	}

	//element-wise matrix multiplication
	private static double[][] elementMatMultiply(double[][] X, double[][] Y) {
		int rowsize = X.length;	
		int colsize = Y[0].length;	
		double[][] Z = new double[rowsize][colsize];		
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				Z[i][j] = X[i][j] * Y[i][j];
			}
		} 
		return Z;
	}
	//scalar matrix multiplication
	private static double[][] scalarMultiply(double[][] X, double alpha) {
		int rowsize = X.length;	
		int colsize = X[0].length;			
		double[][] Z = new double[rowsize][colsize];		
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				Z[i][j] = alpha * X[i][j];
			}
		} 
		return Z;
	}
    
	//softmax that returns a scalar   	
	private static double softmax(double[] b, int i) {
        double denominator = 0;
        for (int j = 0; j < b.length; j++) {
            denominator += Math.exp(b[j]);
        }
        return Math.exp(b[i]) / denominator;
    }
	
	//softmax that returns a vector
	private static double[] vecSoftmax(double[] b) {
		double[] softmaxb = new double[b.length];
		for (int i = 0; i < b.length; i++) {
			softmaxb[i] = softmax(b, i);
		}
		return softmaxb;
	}

	//softmax that returns a matrix
	private static double[][] matrixSoftmax(double[][] B) {
		double[][] softmaxB = new double[B.length][B[0].length];
		for (int i = 0; i < B.length; i++) {
			softmaxB[i] = vecSoftmax(B[i]);
		}
		return softmaxB;
	}
	
	//logistic function
	private static double sigmoid(double z) {
		return 1.0/(1 +  Math.exp(-1.0*z));
	}
	
	//logistic function for vectors	
	private static double[] vecSigmoid(double[] Z){
		double[] sigZ = new double[Z.length];
		for (int i = 0; i < Z.length; i++) {
			sigZ[i] = sigmoid(Z[i]);
		}
		return sigZ;
	}

	//function that takes a vector and returns a vector with an extra 1 at the beginning
	private static double[] pad1(double[] h) {
		double[] hplus1 = new double[h.length + 1];
		System.arraycopy(h, 0, hplus1, 1, h.length);
		hplus1[0] = 1;
		return hplus1;
	}

	//function that takes a vector and returns a vector with one less dimension (first index is tossed)
	private static double[] removeFirstIndex(double[] h) {
		double[] hminus1 = new double[h.length -1];
		for(int i = 1; i < h.length; i++) {
			hminus1[i-1] = h[i];
		}
		return hminus1;
	}

	//function that takes a matrix and returns a matrix with an extra column of 1's at beginning
	private static double[][] pad1Matrix(double[][] H) {
		double[][] HplusOnes = new double[H.length][H[0].length + 1];
		for (int i = 0; i < H.length; i++) {
			HplusOnes[i] = pad1(H[i]);
		}
		return HplusOnes;
	}

	//logistic function for matrices	
	private static double[][] matSigmoid(double[][] Z){
		double[][] sigZ = new double[Z.length][Z[0].length];
		for (int i = 0; i < Z.length; i++) {
			sigZ[i] = vecSigmoid(Z[i]);
		}
		return sigZ;
	}
	
//administrative functions--------------------------------------------------------------
/* Program output - print each iteration of points' Y values to standard out and
    * The average loss of the Dev obj. to standard error. Example format below:
   	Iter 0001: devObj=3.23233
	train [0.72 0.28] [0.85 0.15] [0.35 0.65]
	dev [0.98 0.02] [0.07 0.93] [0.68 0.32] [0.10 0.90] */
    public static void printAssignments(String printstyle, double devObjectiveValue) {
        System.err.println(String.format("Iter %04d: devObj=%.5f", iterCounter, devObjectiveValue));
        System.err.flush();
		
		if (printstyle.equals("verbose")) {
			System.out.print("train ");
		    for (int p = 0; p < ntrain; p++) {
				System.out.print("[");			
				for (int c = 0; c < C; c++) {		            
					System.out.printf("%.2f ", trainY[p][c]);
				}
				System.out.print("] ");
		    }
		    System.out.println();
		}

        System.out.print("dev ");
        for (int p = 0; p < ndev; p++) {
			System.out.print("[");			
			for (int c = 0; c < C; c++) {		            
				System.out.printf("%.2f ", devY[p][c]);
			}
			System.out.print("] ");
        }
        System.out.println();
        System.out.flush();
    }

    //readDataFile - reads datafiles to return 2d arrays of points or Label vectors.
    private static double[][] readDataFile(String file, String type, int expectedNumInputs, int dimension) {
		try {
			double[][] inputs;            
			double[] datapoint;			
			Scanner scanner = new Scanner(new File(file));
			if (type.equals("x")) {       		
				inputs = new double[expectedNumInputs][dimension + 1];
			} else {
				inputs = new double[expectedNumInputs][dimension];
			}
		    int nCount = 0;
		    while (scanner.hasNextLine()) {
				int dCount = 0;		        
				Scanner lineScan = new Scanner(scanner.nextLine());
		        
				if (type.equals("x")) {       		
					datapoint = new double[dimension + 1];					
					datapoint[0] = 1;		        
					dCount += 1;
				} else {
					datapoint = new double[dimension];
				}		
				while (lineScan.hasNext()) {
		            if (dCount > dimension) {
		                quitProgram("Error: Read too many input features for a single input.");
		            }
		            datapoint[dCount++] = lineScan.nextDouble();
		        }
		        inputs[nCount] = datapoint;
				nCount++;
		    }
			if (nCount != expectedNumInputs) {
		        quitProgram("Error: Read too many or too few inputs for a single input.");
		    }
		    return inputs;
		}
		catch(FileNotFoundException ex) {
            System.err.println(String.format("Error: file not found: %s", file));
            System.exit(1);
        }
		return null;
    }

    //Quits program with exit value 1.
    private static void quitProgram(String msg) {
        System.err.println(msg);
        System.exit(1);
    }
//--------------------------------------------------------------------------------
}//end class


