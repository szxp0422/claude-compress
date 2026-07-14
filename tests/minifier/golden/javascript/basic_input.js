// Basic JS: functions, loops, conditionals, closures.
// This comment should be stripped.

function addNumbers(firstValue, secondValue) {
    // Add two numbers
    const resultValue = firstValue + secondValue;
    return resultValue;
}

function computeSum(numbersList) {
    /* Compute sum of an array */
    let totalAccumulator = 0;
    for (const currentItem of numbersList) {
        totalAccumulator = totalAccumulator + currentItem;
    }
    return totalAccumulator;
}

function checkPositive(inputValue) {
    let statusFlag;
    if (inputValue > 0) {
        statusFlag = true;
    } else {
        statusFlag = false;
    }
    return statusFlag;
}

// Top-level calls
console.log(addNumbers(3, 4));
console.log(computeSum([1, 2, 3, 4, 5]));
console.log(checkPositive(-1));
console.log(checkPositive(5));
