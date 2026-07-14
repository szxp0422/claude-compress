# Tests: exec/eval disables renaming in affected scope.
# Global/nonlocal properly excluded from function locals.

global_state = []

GLOBAL_COUNTER = 0


def safe_function(first_arg, second_arg):
    """This function has no exec/eval — renaming is safe."""
    local_sum = first_arg + second_arg
    local_product = first_arg * second_arg
    return local_sum, local_product


def unsafe_function(code_string):
    """This function uses eval — renaming must be DISABLED for this scope."""
    local_var = 10
    result_value = eval(code_string)  # noqa: S307
    return result_value


def uses_global():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    local_copy = GLOBAL_COUNTER
    return local_copy


def uses_nonlocal():
    outer_state = [0]

    def inner_incrementer(delta):
        nonlocal outer_state
        outer_state = [outer_state[0] + delta]
        return outer_state[0]

    return inner_incrementer


if __name__ == "__main__":
    print(safe_function(3, 4))       # (7, 12)
    print(unsafe_function("1 + 2"))  # 3
    print(uses_global())             # 1
    print(uses_global())             # 2

    inc = uses_nonlocal()
    print(inc(5))   # 5
    print(inc(3))   # 8
