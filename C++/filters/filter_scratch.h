#pragma once

#include "../libraries/eigen-5.0.0/Eigen/Dense"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Measurement {
    double t;   // seconds
    double ax;  // m/s^2
    double ay;  // m/s^2
    double az;  // m/s^2
};

// ============================================================
// filter_scratch
//
// Build your own filter from scratch. At minimum it needs:
//   - Initialization step  -> constructor
//   - Predict step          -> predict()
//   - Update step            -> update()
//
// state is assumed to be:
//   [ position_x, velocity_x, position_y, velocity_y, position_z, velocity_z ]
// ============================================================

class filter_scratch {
public:
    filter_scratch();

    // Predict step
    void predict(const Measurement& measurement);

    // Update step
    void update(const Measurement& measurement);

    Eigen::VectorXd state;


    
    std::vector<Measurement> load_measurements(const std::string& csv_path) {
        std::vector<Measurement> data;
        std::ifstream file(csv_path);

        if (!file.is_open()) {
            std::cerr << "ERROR: could not open " << csv_path << std::endl;
            return data;
        }

        std::string line;
        bool first_line = true;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            // Skip a header row if the first field isn't numeric
            if (first_line) {
                first_line = false;
                if (!line.empty() && !(std::isdigit(line[0]) || line[0] == '-' || line[0] == '+' || line[0] == '.')) {
                    continue;
                }
            }

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;

            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    row.push_back(0.0);
                }
            }

            if (row.size() >= 4) {
                Measurement m;
                m.t = row[0];
                m.ax = row[1];
                m.ay = row[2];
                m.az = row[3];
                data.push_back(m);
            }
        }

        return data;
    }



};

