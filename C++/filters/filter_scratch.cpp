#include "filter_scratch.h"

#include "../libraries/eigen-5.0.0/Eigen/Dense"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// MAKE SURE YOU CHANGE THE .h file as well!!!! 


// initialize the filter with a zero state and any other member variables you need
filter_scratch::filter_scratch() {
    std::cout << "initialize run" << std::endl;

    state = Eigen::VectorXd::Zero(6);
}

// Predict Step 
void filter_scratch::predict(const Measurement& measurement) {
    std::cout << "predict run" << std::endl;
    return; 
}

// Update Step
void filter_scratch::update(const Measurement& measurement) {
    std::cout << "update run" << std::endl;

    return;
}

int main() {

    filter_scratch filter;
    const std::string csv_path = "../static/SAWA_Decimate.csv";
    std::vector<Measurement> data = filter.load_measurements(csv_path);

    if (data.empty()) {
        std::cerr << "No data loaded from " << csv_path << std::endl;
        return 1;
    }

    std::vector<Eigen::VectorXd> state_history;
    state_history.reserve(data.size());

    for (const auto& measurement : data) {
        filter.predict(measurement);
        filter.update(measurement);
        state_history.push_back(filter.state);
    }

    std::ofstream out("filter_output.csv");
    out << "t,pos_x,vel_x,pos_y,vel_y,pos_z,vel_z,meas_ax,meas_ay,meas_az\n";
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& s = state_history[i];
        const auto& m = data[i];
        out << m.t << "," << s(0) << "," << s(1) << "," << s(2) << ","
            << s(3) << "," << s(4) << "," << s(5) << "," << m.ax << ","
            << m.ay << "," << m.az << "\n";
    }

    out.close();

    std::cout << "Ran filter over " << data.size()
              << " measurements. Results written to filter_output.csv"
              << std::endl;

    return 0;
}