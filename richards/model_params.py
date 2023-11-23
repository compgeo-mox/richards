import pygeon
import porepy
import sympy as sp
import numpy as np

class Model_Data:
    """
    Simple class that stores the parameters required to perform a Richard simulation (from the phisical parameters, to the solver ones)
    """
    def __init__(self, 
                 theta_r, theta_s, alpha, n, K_s, 
                 T, num_steps) -> None:
        self.theta_r = theta_r
        self.theta_s = theta_s

        self.alpha = alpha

        self.n = n
        self.K_s = K_s

        self.T    = T

        self.theta_m = theta_s
        self.m = 1 - 1/n

        self.num_steps = num_steps
        self.dt   = (T-0) / self.num_steps

        self.derivative_theta = list()
        self.derivative_K = list()
        self.derivative_K_inv = list()

        self.h_var = sp.Symbol('h', negative=True)
        self.z_var = sp.Symbol('z', negative=True)

        self.theta_expression = self.theta_r + (self.theta_s - self.theta_r) / (1 + (-self.alpha * (self.h_var - self.z_var)) ** self.n) ** self.m
        
        self.effective_saturation = (self.theta_expression - theta_r) / (theta_s - theta_r)
        self.hydraulic_conductivity_expression = K_s * (self.effective_saturation ** 0.5) * ( 1 - (1 - self.effective_saturation ** (1 / self.m)) ** self.m ) ** 2


    def __theta_setup(self, order):
        """
        Prepare the theta and its derivatives up to the specified order
        """

        if len(self.derivative_theta) == 0:
            self.derivative_theta.append(
                sp.lambdify([self.h_var, self.z_var],
                            sp.Piecewise(
                                (self.theta_expression, self.h_var < self.z_var),
                                (self.theta_s, True)
                            ), 'numpy')
            )

        fixed_len = len(self.derivative_theta)
        
        for od in range(order - fixed_len + 1):
            self.derivative_theta.append( 
                sp.lambdify([self.h_var, self.z_var], 
                            sp.Piecewise(
                                (sp.diff(self.theta_expression, self.h_var, fixed_len + od), self.h_var < self.z_var),
                                (0, True)
                            ), 'numpy') )

    def __hydraulic_setup(self, order):
        """
        Prepare the hydraulic conductivity coefficient and its derivatives up to the specified order
        """

        if len(self.derivative_K) == 0:
            self.derivative_K.append(
                sp.lambdify([self.h_var, self.z_var],
                            sp.Piecewise(
                                (self.hydraulic_conductivity_expression, self.h_var < self.z_var),
                                (self.K_s, True)
                            ), 'numpy')
            )

        fixed_len = len(self.derivative_K)
        
        for od in range( order - fixed_len + 1):
            self.derivative_K.append(
                sp.lambdify([self.h_var, self.z_var],
                            sp.Piecewise(
                                (sp.diff(self.hydraulic_conductivity_expression, self.h_var, fixed_len + od), self.h_var < self.z_var),
                                (0, True)
                            ), 'numpy')
            )

    def __inv_hydraulic_setup(self, order):
        """
        Prepare the inverse hydraulic conductivity coefficient and its derivatives up to the specified order
        """

        if len(self.derivative_K_inv) == 0:
            self.derivative_K_inv.append(
                sp.lambdify([self.h_var, self.z_var],
                            sp.Piecewise(
                                (1 / self.hydraulic_conductivity_expression, self.h_var < self.z_var),
                                (1 / self.K_s, True)
                            ), 'numpy')
            )

        fixed_len = len(self.derivative_K_inv)
        
        for od in range( order - fixed_len + 1):
            self.derivative_K_inv.append(
                sp.lambdify([self.h_var, self.z_var],
                            sp.Piecewise(
                                (sp.diff(1 / self.hydraulic_conductivity_expression, self.h_var, fixed_len + od), self.h_var < self.z_var),
                                (0, True)
                            ), 'numpy')
            )


    def theta(self, h, z, order = 0):
        """
        It will return theta (or one of its derivatives) evaluated in the cell centers.
        """
        if len(self.derivative_theta) <= order:
            self.__theta_setup(order)

        return self.derivative_theta[order](h, z)


    def hydraulic_conductivity_coefficient(self, h, z, order = 0):
        """
        It will return the hydraulic conductivity (or one of its derivatives) evaluated in the cell centers.
        """
        if len(self.derivative_K) <= order:
            self.__hydraulic_setup(order)

        return self.derivative_K[order](h, z)


    def inverse_hydraulic_conductivity_coefficient(self, h, z, order = 0):
        """
        It will return the inverse hydraulic conductivity (or one of its derivatives) evaluated in the cell centers.
        """
        if len(self.derivative_K_inv) <= order:
            self.__inv_hydraulic_setup(order)

        return self.derivative_K_inv[order](h, z)