import pygeon
import porepy
import sympy as sp
import numpy as np

class Model_Data:
    def __init__(self, theta_r, theta_s, alpha, n, K_s, T, num_steps) -> None:
        self.theta_r = theta_r
        self.theta_s = theta_s

        self.alpha = alpha

        self.n = n
        self.K_s = K_s

        self.T    = T

        self.h_s = 0
        self.theta_m = theta_s
        self.m = 1 - 1/n

        self.num_steps = num_steps
        self.dt   = (T-0) / self.num_steps

        self.derivative_theta = list()
        self.derivative_K = list()
        self.derivative_K_inv = list()
        
        self.psi_var = sp.Symbol('psi', negative=True)
        self.theta_expression = self.theta_r + (self.theta_s - self.theta_r) / (1 + (-self.alpha * self.psi_var) ** self.n) ** self.m
        
        self.effective_saturation = (self.theta_expression - theta_r) / (theta_s - theta_r)
        self.hydraulic_conductivity_expression = K_s * (self.effective_saturation ** 0.5) * ( 1 - (1 - self.effective_saturation ** (1 / self.m)) ** self.m ) ** 2


    def __theta_setup(self, order):
        fixed_len = len(self.derivative_theta)

        for od in range( order - fixed_len + 1):
            self.derivative_theta.append( sp.lambdify(self.psi_var, sp.diff(self.theta_expression, self.psi_var, fixed_len + od), 'numpy') )

    def __hydraulic_setup(self, order):
        fixed_len = len(self.derivative_K)
        
        for od in range( order - fixed_len + 1):
            self.derivative_K.append( sp.lambdify(self.psi_var, sp.diff(self.hydraulic_conductivity_expression, self.psi_var, fixed_len + od), 'numpy') )

    def __inv_hydraulic_setup(self, order):
        fixed_len = len(self.derivative_K_inv)
        
        for od in range( order - fixed_len + 1):
            self.derivative_K_inv.append( sp.lambdify(self.psi_var, sp.diff(1 / self.hydraulic_conductivity_expression, self.psi_var, fixed_len + od), 'numpy') )

    
    def __internal_theta(self, psi, order):
        mask = psi < self.h_s

        if order == 0:
            res = np.ones_like(psi) * self.theta_s
        else:
            res = np.zeros_like(psi)

        res[mask] = self.derivative_theta[order](psi[mask])

        return res
    
    
    def __internal_hydraulic(self, psi, order):
        mask = psi < self.h_s

        if order == 0:
            res = np.ones_like(psi) * self.K_s
        else:
            res = np.zeros_like(psi)

        res[mask] = self.derivative_K[order](psi[mask])

        return res
    
    
    def __internal_inv_hydraulic(self, psi, order):
        mask = psi < self.h_s

        if order == 0:
            res = np.ones_like(psi) / self.K_s
        else:
            res = np.zeros_like(psi)

        res[mask] = self.derivative_K_inv[order](psi[mask])

        return res


    def theta(self, psi, order = 0):
        if len(self.derivative_theta) <= order:
            self.__theta_setup(order)

        return self.__internal_theta(psi, order)


    def hydraulic_conductivity_coefficient(self, psi, order = 0):
        if len(self.derivative_K) <= order:
            self.__hydraulic_setup(order)

        return self.__internal_hydraulic(psi, order)


    def inverse_hydraulic_conductivity_coefficient(self, psi, order = 0):
        if len(self.derivative_K_inv) <= order:
            self.__inv_hydraulic_setup(order)

        return self.__internal_inv_hydraulic(psi, order)