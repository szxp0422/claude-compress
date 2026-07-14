// Tests: closures, nested scopes, var hoisting, let/const block scope.

function makeCounter(startValue) {
    let currentCount = startValue;

    function increment(stepSize) {
        currentCount = currentCount + stepSize;
        return currentCount;
    }

    function reset() {
        currentCount = startValue;
    }

    return { increment, reset };
}

function makeMultiplier(factorValue) {
    return function(inputNumber) {
        const productResult = inputNumber * factorValue;
        return productResult;
    };
}

const arrowMultiplier = (factorValue) => (inputNumber) => inputNumber * factorValue;

function varHoistingTest(limitValue) {
    // var is hoisted to function scope
    var collectedResults = [];
    for (var loopIndex = 0; loopIndex < limitValue; loopIndex++) {
        var doubledValue = loopIndex * 2;
        collectedResults.push(doubledValue);
    }
    return collectedResults;
}

const counter = makeCounter(10);
console.log(counter.increment(1));   // 11
console.log(counter.increment(2));   // 13
counter.reset();
console.log(counter.increment(0));   // 10

const double = makeMultiplier(2);
const triple = makeMultiplier(3);
console.log(double(5));   // 10
console.log(triple(5));   // 15

const arrowDouble = arrowMultiplier(2);
console.log(arrowDouble(7));  // 14

console.log(varHoistingTest(4));  // [0, 2, 4, 6]
