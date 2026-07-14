# Tests: module-level names must NOT be renamed (they may be imported).
# Class-level attributes must NOT be renamed.

PUBLIC_CONSTANT = 42


class Calculator:
    """A simple calculator class."""

    multiplier = 10  # class attribute — accessible as Calculator.multiplier

    def __init__(self, initial_value):
        self.value = initial_value

    def add(self, amount_to_add):
        local_result = self.value + amount_to_add
        self.value = local_result
        return local_result

    def multiply_by_class_attr(self):
        scaled_value = self.value * Calculator.multiplier
        return scaled_value


def use_calculator():
    calc_instance = Calculator(PUBLIC_CONSTANT)
    first_result = calc_instance.add(8)
    second_result = calc_instance.multiply_by_class_attr()
    return first_result, second_result


if __name__ == "__main__":
    r1, r2 = use_calculator()
    print(r1)   # 50
    print(r2)   # 500
