#include <iostream>
#include <string>
#include <vector>
#include <array>
#include "../libraries/eigen-5.0.0/Eigen/Dense"

// Functions (START AT MAIN)


int add (int a, int b){

    return a + b; 

}

void print_the_best_team() {

    std::cout << "the best team is GNC" << std::endl; 

}

struct Position {

    double x; 
    double y; 
    double z; 


}; // <- this is important 

class Rocket {

    public: 
        Rocket (double num_fins){

            num_fins_ = num_fins;
        }

        void name_fins() {
            std::cout << "The number of fins are " << num_fins_ << std::endl;  
        }

    private: 
        int num_fins_; 


}; 



int main (){  // EVERY C++ program needs a main function. This is what runs when you compile your program

    // vars 
    int x = 0; 
    double y = 3.14; 
    char c = 'I'; 
    bool ISS_is_goated = true; 
    const double PI = 3.14159; 

    // iostream 

    std::cout << "Hello ISS!" << std::endl;
    bool gnc_goated; 
    std::cout << "enter if GNC is goated" << std::endl; 
    std::cin >> gnc_goated; 

    // operators 

    int a = 5; 
    int b = 8;

    int integer_dev = a/b; 
    double decimal_result = (double)a / b; 
    double intresting_result = (double)(a/b);

    // arithmetic 
    
    int sum = a+b; 
    int diff = a-b; 
    int product = a*b;
    int remainder = a%b; 

    // increment and decrement 

    int count = 5; 
    count++; 
    count--; 

    // compounding 

    count += 2;
    count -= 2; 
    count /= 2; 
    count *= 2; 

    // Comparisons:

    bool equal = (a == b);
    bool not_equal = (a != b);
    bool greater = (a > b);
    bool less = (a < b);
    bool greater_equal = (a >= b);
    bool less_equal = (a <= b);

    // Logical operators:

    bool result1 = (a > 0 && b > 0);  // AND
    bool result2 = (a > 0 || b > 0);  // OR
    bool result3 = !(a == b);         // NOT

    // Strings (less useful for GNC but good to know)

    std::string subteam = "gnc";
    std::string the_truth = subteam + " is the goated team!"; 
    
    std::cout << the_truth << std::endl;
    std::cout << the_truth.length() << std::endl;
    std::cout << the_truth.size() << std::endl;
    
    // characters 

    char guidance = subteam[0]; 

    // arrays and std (we use Eigen matrices on the GNC team but it is good to know)

    int numbers[4] = {1,2,3,6}; 

    std::array<int, 3> fixed_array = {1,2,3}; 
    
    // dynamic
    std::vector<int> values; 
    values.push_back(10);
    values.push_back(20);
    values.push_back(30);

    std::cout << "Vector size: " << values.size() << std::endl;
    std::cout << "First element: " << values[0] << std::endl;

    // Loops 

    // For loop:
    for (int i = 0; i < 5; ++i)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // While loop:
    int n = 5;

    while (n > 0)
    {
        std::cout << n << " ";
        n--;
    }
    std::cout << std::endl;

    // Range-based for loop:
    for (int value : numbers)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Conditionals 

        int score = 85;

    if (score >= 90)
    {
        std::cout << "A" << std::endl;
    }
    else if (score >= 80)
    {
        std::cout << "B" << std::endl;
    }
    else
    {
        std::cout << "C or below" << std::endl;
    }

    // functions utilized 

    int addition = add(a, b); 

    // structs 

    Position p{1.0, 2.0, 4.0};

    std::cout << "Position: " << p.x << ", " << p.y << p.z << std::endl;

    // class 

    Rocket Aether = Rocket(3); 
    Aether.name_fins(); 

    // pointers and reference ...not as useful for GNC but good to know. SW will use this much more 
    
    int value = 42;

    // Reference: another name for the same variable.
    int& reference = value;

    // Pointer: stores the address of a variable.
    int* pointer = &value;

    // Dereference pointer to access/change the value.
    *pointer = 100;

    std::cout << value << std::endl;
    std::cout << reference << std::endl;
    std::cout << *pointer << std::endl;


    // Eigen 
     
    Eigen::Matrix3d A;
    Eigen::VectorXd v(3);
    Eigen::Vector3d w;

    Eigen::MatrixXd B(2, 3);

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Vector3d zeros = Eigen::Vector3d::Zero();

    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    v << 1.0, 2.0, 3.0;

    A(0, 0) = 99;

    double matrix_value = A(1, 2);

    Eigen::Matrix3d fixed_matrix;

    Eigen::MatrixXd dynamic_matrix(4, 4);

    Eigen::Vector3d fixed_vector;

    Eigen::VectorXd dynamic_vector(4);

    fixed_matrix = Eigen::Matrix3d::Identity();

    fixed_vector = Eigen::Vector3d::Zero();

    Eigen::Matrix3d add_result = A + I;

    Eigen::Matrix3d multiply_result = A * I;

    Eigen::Vector3d vector_result = A * v;

    Eigen::Matrix3d transposed = A.transpose();


}
