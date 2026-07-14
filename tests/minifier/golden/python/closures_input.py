# Tests: closures, shadowing, nested functions, nonlocal.

def make_counter(start_value):
    current_count = start_value

    def increment(step_size):
        nonlocal current_count
        current_count = current_count + step_size
        return current_count

    def reset():
        nonlocal current_count
        current_count = start_value

    return increment, reset


def make_multiplier(factor_value):
    def multiply(input_number):
        # factor_value is a free variable from enclosing scope
        product_result = input_number * factor_value
        return product_result
    return multiply


def shadowing_test(outer_x):
    result_list = []
    for outer_x in range(outer_x):  # outer_x shadows the parameter here
        inner_val = outer_x * 2
        result_list.append(inner_val)
    return result_list


if __name__ == "__main__":
    inc, rst = make_counter(10)
    print(inc(1))   # 11
    print(inc(2))   # 13
    rst()
    print(inc(0))   # 10

    double = make_multiplier(2)
    triple = make_multiplier(3)
    print(double(5))   # 10
    print(triple(5))   # 15

    print(shadowing_test(4))  # [0, 2, 4, 6]
