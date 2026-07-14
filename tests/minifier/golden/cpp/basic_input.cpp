// Basic C++: class, methods, free functions, templates

#include <iostream>
#include <string>

/* Global constant — NOT renamed */
const int MAX_ITERATIONS = 1000;

// Free function — parameters and locals renamed
int add_integers(int first_value, int second_value) {
    int local_sum = first_value + second_value;
    return local_sum;
}

// Template function — type parameter T not renamed (type_identifier)
template <typename T>
T clamp_value(T input_val, T lower_bound, T upper_bound) {
    T clamped_result = input_val;
    if (clamped_result < lower_bound) {
        clamped_result = lower_bound;
    }
    if (clamped_result > upper_bound) {
        clamped_result = upper_bound;
    }
    return clamped_result;
}

class Counter {
    // Private member — field_identifier, not renamed
    int current_count;
    int step_size;

public:
    Counter(int initial_value, int increment_step) {
        current_count = initial_value;
        step_size = increment_step;
    }

    void increment() {
        int new_value = current_count + step_size;
        current_count = new_value;
    }

    int get_count() const {
        int result_copy = current_count;
        return result_copy;
    }

    void reset(int reset_value) {
        current_count = reset_value;
    }
};

int main() {
    int sum_result = add_integers(10, 20);
    std::cout << sum_result << std::endl;

    int clamped = clamp_value(15, 0, 10);
    std::cout << clamped << std::endl;

    Counter my_counter(0, 2);
    my_counter.increment();
    my_counter.increment();
    my_counter.increment();
    std::cout << my_counter.get_count() << std::endl;

    return 0;
}
