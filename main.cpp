#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

class ParallelCalculator {
private:
    int rank, size;
    std::vector<double> numbers;
    
    enum Operation {
        ADD = 1,
        MULTIPLY = 2,
        POWER = 3,
        FACTORIAL = 4,
        AVERAGE = 5
    };

    double calculateFactorial(int n) {
        if (n <= 1) return 1;
        double result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

public:
    ParallelCalculator(int r, int s) : rank(r), size(s) {}

    void readNumbers() {
        if (rank == 0) {
            int count;
            std::cout << "How many numbers do you want to enter? ";
            std::cin >> count;
            
            numbers.resize(count);
            std::cout << "Enter " << count << " numbers:\n";
            for (int i = 0; i < count; i++) {
                std::cout << "Number " << (i + 1) << ": ";
                std::cin >> numbers[i];
            }
        }

        int count = numbers.size();
        MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            numbers.resize(count);
        }

        MPI_Bcast(numbers.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void displayMenu() {
        if (rank == 0) {
            std::cout << "\nAvailable Operations:\n";
            std::cout << "1. Add Numbers\n";
            std::cout << "2. Multiply Numbers\n";
            std::cout << "3. Raise All Numbers to Power\n";
            std::cout << "4. Calculate Factorial of Numbers\n";
            std::cout << "5. Calculate Average\n";
            std::cout << "6. Enter New Numbers\n";
            std::cout << "7. Exit\n";
            std::cout << "Choose operation: ";
        }
    }

    void processOperation(int operation) {
        int local_size = numbers.size() / size;
        int remainder = numbers.size() % size;
        int start_idx = rank * local_size + std::min(rank, remainder);
        if (rank < remainder) local_size++;

        double local_result = 0;

        if (operation == POWER) {
            double power_exponent = 2;
            if (rank == 0) {
                std::cout << "Enter power value: ";
                std::cin >> power_exponent;
            }
            MPI_Bcast(&power_exponent, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<double> local_results;
            for (int i = 0; i < local_size && start_idx + i < numbers.size(); i++) {
                double result = std::pow(numbers[start_idx + i], power_exponent);
                local_results.push_back(result);
            }

            if (rank == 0) {
                std::vector<int> recvcounts(size);
                std::vector<int> displs(size);
                
                for (int i = 0; i < size; i++) {
                    recvcounts[i] = numbers.size() / size;
                    if (i < remainder) recvcounts[i]++;
                    displs[i] = (i > 0) ? displs[i-1] + recvcounts[i-1] : 0;
                }

                std::vector<double> all_results(numbers.size());
                MPI_Gatherv(local_results.data(), local_results.size(), MPI_DOUBLE,
                           all_results.data(), recvcounts.data(), displs.data(),
                           MPI_DOUBLE, 0, MPI_COMM_WORLD);

                for (int i = 0; i < numbers.size(); i++) {
                    std::cout << numbers[i] << " ^ " << power_exponent 
                            << " = " << all_results[i] << "\n";
                }
            } else {
                MPI_Gatherv(local_results.data(), local_results.size(), MPI_DOUBLE,
                           nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            return;
        }
        else if (operation == FACTORIAL) {
            for (int i = 0; i < local_size && start_idx + i < numbers.size(); i++) {
                double result = calculateFactorial(static_cast<int>(numbers[start_idx + i]));
                if (rank == 0) {
                    std::cout << numbers[start_idx + i] << "! = " << result << "\n";
                }
            }
            return;
        }
        else {
            switch (operation) {
                case ADD:
                    local_result = 0;
                    for (int i = 0; i < local_size && start_idx + i < numbers.size(); i++) {
                        local_result += numbers[start_idx + i];
                    }
                    break;

                case MULTIPLY:
                    local_result = 1;
                    for (int i = 0; i < local_size && start_idx + i < numbers.size(); i++) {
                        local_result *= numbers[start_idx + i];
                    }
                    break;

                case AVERAGE:
                    local_result = 0;
                    for (int i = 0; i < local_size && start_idx + i < numbers.size(); i++) {
                        local_result += numbers[start_idx + i];
                    }
                    break;
            }

            double global_result;
            MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, 
                      operation == MULTIPLY ? MPI_PROD : MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                switch (operation) {
                    case ADD:
                        std::cout << "Sum of numbers = " << global_result << "\n";
                        break;
                    case MULTIPLY:
                        std::cout << "Product of numbers = " << global_result << "\n";
                        break;
                    case AVERAGE:
                        std::cout << "Average of numbers = " << global_result / numbers.size() << "\n";
                        break;
                }
            }
        }
    }

    void run() {
        readNumbers();
        
        while (true) {
            int choice = 0;
            
            if (rank == 0) {
                displayMenu();
                std::cin >> choice;
            }

            MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (choice == 7) break;
            
            if (choice == 6) {
                readNumbers();
                continue;
            }

            if (choice >= 1 && choice <= 5) {
                processOperation(choice);
            } else if (rank == 0) {
                std::cout << "Invalid choice!\n";
            }
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ParallelCalculator calculator(rank, size);
    calculator.run();

    MPI_Finalize();
    return 0;
}