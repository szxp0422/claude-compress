// Basic TypeScript: functions, generics, classes, type annotations

interface Shape {
    area(): number;
    perimeter(): number;
}

type Point = { x: number; y: number };

function addValues(firstNumber: number, secondNumber: number): number {
    const localResult: number = firstNumber + secondNumber;
    return localResult;
}

const multiplyValues = (leftOperand: number, rightOperand: number): number => {
    const product: number = leftOperand * rightOperand;
    return product;
};

function computeDistance(pointA: Point, pointB: Point): number {
    const deltaX: number = pointA.x - pointB.x;
    const deltaY: number = pointA.y - pointB.y;
    return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
}

class Rectangle implements Shape {
    private width: number;
    private height: number;

    constructor(rectWidth: number, rectHeight: number) {
        this.width = rectWidth;
        this.height = rectHeight;
    }

    area(): number {
        const surfaceArea: number = this.width * this.height;
        return surfaceArea;
    }

    perimeter(): number {
        const totalPerimeter: number = 2 * (this.width + this.height);
        return totalPerimeter;
    }
}

// Generics
function identity<T>(inputValue: T): T {
    const outputValue: T = inputValue;
    return outputValue;
}

const rect = new Rectangle(3, 4);
console.log(addValues(10, 20));
console.log(multiplyValues(3, 7));
console.log(rect.area());
console.log(rect.perimeter());
console.log(identity(42));
