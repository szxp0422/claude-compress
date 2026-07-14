// Basic Java: class, methods, local variables, loops

public class BasicInput {

    // Field — NOT renamed (accessible via reflection)
    private int instanceState;

    public BasicInput(int initialState) {
        this.instanceState = initialState;
    }

    // Method parameters and locals ARE renamed
    public int addNumbers(int firstNumber, int secondNumber) {
        int localResult = firstNumber + secondNumber;
        this.instanceState = localResult;
        return localResult;
    }

    public int computeSum(int[] inputArray) {
        int totalAccumulator = 0;
        for (int loopIndex = 0; loopIndex < inputArray.length; loopIndex++) {
            int currentElement = inputArray[loopIndex];
            totalAccumulator = totalAccumulator + currentElement;
        }
        return totalAccumulator;
    }

    public String processItems(String[] itemList) {
        StringBuilder resultBuilder = new StringBuilder();
        for (String currentItem : itemList) {
            resultBuilder.append(currentItem);
            resultBuilder.append(",");
        }
        return resultBuilder.toString();
    }

    public int safeOperation(int inputValue) {
        try {
            int computedResult = inputValue * 2;
            if (computedResult < 0) {
                throw new IllegalArgumentException("negative");
            }
            return computedResult;
        } catch (IllegalArgumentException exceptionObj) {
            int fallbackValue = 0;
            return fallbackValue;
        }
    }

    public static void main(String[] args) {
        BasicInput calculator = new BasicInput(0);

        int sumResult = calculator.addNumbers(3, 4);
        System.out.println(sumResult);

        int[] numbersArray = {10, 20, 30, 40};
        int totalResult = calculator.computeSum(numbersArray);
        System.out.println(totalResult);

        String[] wordsArray = {"hello", "world"};
        String joinedString = calculator.processItems(wordsArray);
        System.out.println(joinedString);

        int safeResult = calculator.safeOperation(21);
        System.out.println(safeResult);
    }
}
