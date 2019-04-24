import java.util.Random;

public class DixieLearner {

    private static int CHOICES = 3; //Possible choices each turn
    private static int CUPS = 15; //Number of cups (or number of coins)
    private static int NONLEARNING_ITERATIONS = 1000; //Iterations to keep going without learning anything

    /**
     * Pits one against the other
     * @param attacker DixieCup array which goes first
     * @Param defender DixieCup array which goes second
     * @return Returns 0 or 1 denoting which side won and the loser's choices in len(COINS+1) array
     */
    public static int[] compete(boolean[][] array1, boolean[][] array2) {
        int[] choices = new int[CUPS+1];
        int i=0;
        while(i < choices.length){
            choices[i] = 0;
            i++;
        }
        int num_coins = CUPS;
        int num_max = CHOICES;
        Random choice = new Random();
        boolean l = false;
        int madechoice=0;
        int loser=0;
        while (num_coins>0){
            boolean iszero = true;
            while (iszero){
                madechoice = choice.nextInt(num_max);
                l = array1[num_coins-1][madechoice];
                if(l == true){
                    iszero=false;
                }
            }
            choices[num_coins]= madechoice+1;
            num_coins= num_coins-choices[num_coins];
            loser=0;
            if(num_coins>0){
                boolean iszero2 = true;
                int choice2=0;
                while(iszero2){
                    choice2=choice.nextInt(num_max);
                    l=array2[num_coins-1][choice2];
                    if(l == true){
                        iszero2=false;
                    }
                loser=1;
                }
                num_coins = num_coins - (choice2+1);
            }
        choices[0]=loser;
        }
        return choices;
    }

    /**
     * Counts the number of trues in a boolean list
     * @param list Boolean list to count trues in
     * @return Returns number of trues in list
     */
    public static int count(boolean[] list) {
        int num = 0;
        for(boolean item : list) {
            if(item) num++;
        }
        return num;
    }

    /**
     * Runs a single competition and learns from the results (only the attacker learns in this implementation)
     *
     * Because Java passes by reference, this modifies the arrays (ew, side effects)
     * @param attacker DixieCup learner to go first
     * @param defender DixieCup learner to go second
     * @return Returns true if something was changed else false
     */
    public static boolean singleLearn(boolean[][] attacker, boolean[][] defender) {
        int[] results = compete(attacker, defender); //Compete against each other
        boolean loser = results[0] != 0; //loser (defender if true, else attacker)
        boolean[][] temp = loser ? defender : attacker; //the loser: defender if loser is true, else attacker
        boolean done = false;
        if(!loser) {
            //This implementation only teaches the attacker
            for(int i = 0; i < temp.length && !done; i++) {
                //Iterate over temp and determine which to modify in back propogation, then change it
                int coinExpenditure = results[i+1]; //Coins used on turn i (starting with 0)
                if(coinExpenditure != 0) {
                    //It doesn't matter if this wasn't a turn
                    if(coinExpenditure <= temp[i].length) {
                        //Just to make sure it fits in the range
                        if(count(temp[i]) > 1) {
                            //Tests if it is not the last possibility and can be modified
                            temp[i][coinExpenditure-1] = false; //remove this possibility
                            done = true; //quit looping because it has been modified
                        } //else do nothing because it has to go up the food chain
                    } //TODO somehow assert that this is true or something went horribly wrong
                    else { System.out.println("Something failed horribly"); }
                } //else do nothing
            }
        } else done = true; //Because this implementation only teaches the attacker
        return done; //If done is still false, then nothing was changed
    }

    public static boolean[][] initDixieCups(int cups, int choices) {
        boolean[][] array = new boolean[choices][cups];
        for(int i = 0; i < choices; i++) {
            for(int j = 0; j < cups; j++) {
                if(i >= j) {
                    //Because the first ones can't be greater than the coins left
                    array[i][j] = true;
                }
            }
        }
        return array;
    }

    public static String arrayString(boolean[][] array) {
        String ret = "";
        for(boolean[] row : array) {
            for(boolean item : row) {
                ret += (item ? 1 : 0) + " ";
            }
            ret += "\n";
        }
        return ret;
    }

    /**
     * A dictionary notation of the array
     * @param array Boolean 2D dixie cup array
     * @return Returns a nice dictionary-notation string for cups
     */
    public static String prettyArrayString(boolean[][] array) {
        String ret = "{\n";
        for(int i = 0; i < array.length; i++) {
            int numChoices = count(array[i]);
            ret += "    " + (i+1) + ": ";
            if(numChoices > 1) {
                ret += "{";
            }
            int innerCount = 0;
            for(int j = 0; j < array[i].length; j++) {
                if(array[i][j]) {
                    innerCount++;
                    ret += (j+1);
                    if(innerCount < numChoices) {
                        ret += ", ";
                    }
                }
            }
            if(numChoices > 1) {
                ret += "}";
            }
            if(i < array.length - 1) {
                ret += ", ";
            }
            ret += "\n";
        }
        ret += "}";
        return ret;
    }

    public static void main(String[] args) {
        if(args.length >= 1 && (args[0].equals("help") || args[0].equals("h"))) {
            System.out.println("\nDixie Cup Learner\n"
                + "Gary Paradise and Hunter Damron\n"
                + "Command line arguments:\n"
                + " * 'cups {number of cups}'\n    to use a different number of cups (default " + CUPS + ")\n"
                + " * 'choices {number of choices}'\n    to use different number of choices on each turn "
                    + "(default " + CHOICES + ")\n"
                + " * 'repeat {number of repeats}'\n    to change number of repeating times (default " 
                    + NONLEARNING_ITERATIONS + ")\n"
                + " * 'help' to view this message\n"
                + "Example: 'java DixieCup cups 4 choices 17 repeat 1000'\n"); //Help msg
        } else {
            String lookFor = ""; //String saying what to look for i.e. "coins", "choices", "repeats"
            for(int i = 0; i < args.length; i++) {
                if(lookFor == "") {
                    switch(args[i]) {
                        case "choices":
                        case "cups":
                        case "repeat":
                            lookFor = args[i];
                           break;
                        default:
                            System.err.println("Unknown attribute " + args[i]);
                            System.exit(1);
                    }
                } else {
                    int value = 0;
                    try {
                        value = Integer.parseInt(args[i]);
                    } catch(NumberFormatException e) {
                        System.err.println("Invalid number argument " + args[i]);
                        System.exit(1);
                    }
                    switch(lookFor) {
                       case "choices":
                            CHOICES = value;
                            break;
                        case "cups":
                            CUPS = value;
                            break;
                      case "repeat":
                            NONLEARNING_ITERATIONS = value;
                            break;
                        default:
                            System.err.println("How did I get here?"); //This shouldn't happen
                            System.exit(1);
                   }
                    lookFor = ""; //Reset lookFor
                }
            }
            boolean[][] array1 = initDixieCups(CHOICES, CUPS);
            boolean[][] array2 = initDixieCups(CHOICES, CUPS);
            int countdown = NONLEARNING_ITERATIONS;
            while(countdown > 0) {
                boolean learnt = singleLearn(array1, array2);
                if(learnt) { countdown--; } //If learnt, you're one step closer to being done
            }
            System.out.println(prettyArrayString(array1)); //Print out at the end
        }
    }
}
