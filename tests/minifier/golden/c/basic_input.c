/* Basic C: functions, local variables, preprocessor */
#include <stdio.h>

#define MAX_SIZE 100
#define SQUARE(x) ((x) * (x))

/* Global variable — NOT renamed */
int global_counter = 0;

int add_numbers(int first_param, int second_param) {
    /* Local variables — safe to rename */
    int local_result = first_param + second_param;
    global_counter = global_counter + 1;
    return local_result;
}

int compute_sum(int array_size, int *data_array) {
    int total_sum = 0;
    int loop_index = 0;
    while (loop_index < array_size) {
        int current_value = data_array[loop_index];
        total_sum = total_sum + current_value;
        loop_index = loop_index + 1;
    }
    return total_sum;
}

int check_positive(int input_value) {
    if (input_value > 0) {
        int result_flag = 1;
        return result_flag;
    } else {
        int result_flag = 0;
        return result_flag;
    }
}

int main(void) {
    int answer = add_numbers(3, 4);
    printf("%d\n", answer);

    int numbers[3] = {10, 20, 30};
    int total = compute_sum(3, numbers);
    printf("%d\n", total);

    int flag = check_positive(5);
    printf("%d\n", flag);

    return 0;
}
