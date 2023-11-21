import numpy as np

# Constructs the K function in the moving domain framework (with the assumption that the "reference" K=I*coeff)
def quick_K_func_eval(chi_x3, chi_eta, grad_eta, coeff = 1):
    return np.array([[             chi_x3,                            -chi_eta * grad_eta],
                     [-chi_eta * grad_eta, (1 + np.power(chi_eta * grad_eta, 2)) / chi_x3]]) * coeff

# Constructs the K function in the moving domain framework (with the assumption that the "reference" K=[[k11, k12], [k21, k22]]*coeff)
def complete_K_func_eval(chi_x3, chi_eta, grad_eta, k11, k12, k21, k22, coeff=1):
    return np.array([[                  chi_x3 * k11,                                                   k12 - chi_eta * grad_eta * k11],
                     [k21 - chi_eta * grad_eta * k11, ( k22 + chi_eta * grad_eta * ( chi_eta * grad_eta * k11 - k12 - k21 ) ) / chi_x3]]) * coeff