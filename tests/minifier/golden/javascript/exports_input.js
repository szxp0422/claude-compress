// Tests: exported names must NOT be renamed.
// module.exports and named exports should preserve their names.

const PUBLIC_CONSTANT = 42;

function publicFunction(inputValue) {
    const localDouble = inputValue * 2;
    return localDouble;
}

function privateHelper(secretValue, multiplier) {
    const localResult = secretValue * multiplier;
    return localResult;
}

function anotherExport(base, exponent) {
    let accumulator = 1;
    for (let loopCount = 0; loopCount < exponent; loopCount++) {
        accumulator = accumulator * base;
    }
    return accumulator;
}

console.log(publicFunction(21));      // 42
console.log(privateHelper(6, 7));     // 42
console.log(anotherExport(2, 6));     // 64

module.exports = { publicFunction, anotherExport, PUBLIC_CONSTANT };
