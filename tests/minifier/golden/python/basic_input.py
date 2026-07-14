# Basic Python: arithmetic, loops, conditionals.
# This comment should be stripped.

def add_numbers(first_value, second_value):
    # Add two numbers together
    result = first_value + second_value
    return result


def compute_sum(numbers_list):
    """Compute sum of a list."""
    total_accumulator = 0
    for current_item in numbers_list:
        total_accumulator = total_accumulator + current_item
    return total_accumulator


def check_positive(input_value):
    if input_value > 0:
        status_flag = True
    else:
        status_flag = False
    return status_flag


if __name__ == "__main__":
    print(add_numbers(3, 4))
    print(compute_sum([1, 2, 3, 4, 5]))
    print(check_positive(-1))
    print(check_positive(5))
