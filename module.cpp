#include "Rcpp.h"
using namespace Rcpp;

#include <iostream>
#include <cmath>
#include <vector>

// Function to simulate initial state x0
std::vector<double> simX0_2(const std::vector<double>& params, int M) {
    double v0 = params[0];
    return Rcpp::as<std::vector<double>>(Rcpp::rnorm(M, 0, std::sqrt(v0)));
}
// Function to simulate state xt given x at time t-1
std::vector<double> simXt_2(const std::vector<double>& x, const std::vector<double>& params, int M) {
    double a = params[0];
    double b = params[1];
    double vx = params[2];
    
    std::vector<double> xt(M);
    for (int i = 0; i < M; i++) {
        xt[i] = Rcpp::rnorm(1, a * x[i] + b, std::sqrt(vx))[0];
    }
    return xt;
}

NumericMatrix Xproposal_0(NumericVector y0, double eps, int d, int M, NumericVector simX0_params, 
                           Function g_inverse) {
    // Extract the parameter v0
    double v0 = simX0_params[0];
    
    // Calculate delta
    double delta = eps / sqrt(2);
    
    // Get z0 using the provided g.inverse function
    NumericVector z0 = g_inverse(y0, simX0_params);
    
    // Generate random normal values for z0_hat
    NumericVector z0_hat = rnorm(d, 0, 1);
    
    // Calculate phi
    NumericVector phi = sqrt(1 - delta * delta) * z0 + delta * z0_hat;
    
    // Create a matrix for z_hat with M-1 rows and d columns
    NumericMatrix z_hat(M - 1, d);
    
    // Fill z_hat with random normal values
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < d; ++j) {
            z_hat(i, j) = R::rnorm(0, 1); // Use R's random normal generator
        }
    }
    
    // Calculate z
    NumericMatrix z(M - 1, d);
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < d; ++j) {
            z(i, j) = phi[j] * sqrt(1 - delta * delta) + delta * z_hat(i, j);
        }
    }
    
    // Create a matrix for y with M-1 rows and d columns
    NumericMatrix y(M - 1, d);
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < d; ++j) {
            y(i, j) = sqrt(v0) * z(i, j); // Compute y
        }
    }
    
    return y; // Return the resulting matrix
}

NumericMatrix Xproposal_t(const NumericVector& x0, 
                           const NumericVector& y0, 
                           const NumericVector& x, 
                           double eps, 
                           int d, 
                           int M, 
                           const NumericVector& simXt_params, 
                           Function g_inverse) {
    // Extract parameters
    double a = simXt_params[0];  
    double b = simXt_params[1];  
    double vx = simXt_params[2];
    double delta = eps / std::sqrt(2);
    
    // Call the R function g_inverse
    NumericVector z0 = g_inverse(x0, y0, simXt_params);
    
    // Generate random normal values for z0_hat
    NumericVector z0_hat = rnorm(d, 0, 1);
    
    // Calculate phi
    NumericVector phi(d);
    for (int i = 0; i < d; ++i) {
        phi[i] = std::sqrt(1 - delta * delta) * z0[i] + delta * z0_hat[i];
    }
    
    // Create a matrix for z_hat with M-1 rows and d columns
    NumericMatrix z_hat(M - 1, d);
    
    // Fill z_hat with random normal values
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < d; ++j) {
            z_hat(i, j) = R::rnorm(0, 1); // Use R's random normal generator
        }
    }
    
    // Create a matrix for y with M-1 rows and d columns
    NumericMatrix y(M - 1, d);
    
    // Calculate the values of y
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < d; ++j) {
            y(i, j) = a * x[j] + b + std::sqrt(vx) * (phi[j] * std::sqrt(1 - delta * delta) + delta * z_hat(i, j));
        }
    }
    
    return y; // Return the resulting matrix
}

// Function equivalent to g.inverse.0 in R, using simX0_params as a vector input
// [[Rcpp::export]]
std::vector<double> g_inverse_0(const std::vector<double>& y0, const std::vector<double>& simX0_params) {
    double vx = simX0_params[0];  // Extract the first element as vx
    std::vector<double> result(y0.size());
    for (size_t i = 0; i < y0.size(); ++i) {
        result[i] = y0[i] / std::sqrt(vx);
    }
    return result;
}

// Function equivalent to g.inverse.1 in R, handling vector inputs and scalar parameters
// [[Rcpp::export]]
std::vector<double> g_inverse_1(const std::vector<double>& x0, 
                                const std::vector<double>& y0, 
                                const std::vector<double>& simXt_params) {
    double a = simXt_params[0];
    double b = simXt_params[1];
    double vx = simXt_params[2];
    
    std::vector<double> result(y0.size());
    for (size_t i = 0; i < y0.size(); ++i) {
        result[i] = (y0[i] - a * x0[i] - b) / std::sqrt(vx);
    }
    return result;
}

// Rcpp module to expose functions
RCPP_MODULE(dlm){
    Rcpp::function("simX0_2", &simX0_2);
    Rcpp::function("simXt_2", &simXt_2);
    Rcpp::function("Xproposal_0", &Xproposal_0);
    Rcpp::function("Xproposal_t", &Xproposal_t);
    Rcpp::function("g_inverse_0", &g_inverse_0);
    Rcpp::function("g_inverse_1", &g_inverse_1);
}
