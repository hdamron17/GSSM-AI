package main;

public class Main {
	public static void main(String args[]) {
		for(int i = 0; i < 1000; i++)
			if(sumOfSqr(i))
				System.out.println("True for " + i);
	}
	
	public static boolean sumOfSqr(int num) {
		for(int i = 0; i <= num; i++)
			for(int j = 0; i + j <= num; j++)
				if(i * i + j * j == num)
					return true;
		return false;
	}
}