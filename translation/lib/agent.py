import numpy as np

class Agent:
    def __init__(self, A, B, P, dt, x_init, zeta_init):
        """
        Constructor for the Agent class.
        
        :param A: State matrix (nxn)
        :param B: Input matrix (nxr)
        :param P: A constant matrix for calculating controller gain matrix (rxr)
        :param dt: Time step
        :param x_init: Initial state vector (nx1)
        :param zeta_init: Initial value for zeta (dix1)
        """
        self.A = A
        self.B = B
        self.K = - np.dot(B.T, P)  # Controller gain matrix (rxn)
        self.Gamma = np.dot(np.dot(P, np.dot(B, B.T)), P)  # Gamma = P * B * B' * P
        
        self.t = 0.0  # Initial time
        self.dt = dt  # Time step
        
        self.x = x_init  # Initial state vector
        self.dx = np.zeros(A.shape[0])  # State rate of change (dx)
        
        self.u = np.zeros(B.shape[1])  # Input vector
        
        self.zeta = zeta_init  # Initial zeta
        self.dzeta = 0  # Rate of change of zeta
    
    def get_state(self):
        """Returns the current state vector."""
        return self.x
    
    def get_zeta(self):
        """Returns the current zeta."""
        return self.zeta
    
    def update_state(self, hz):
        """
        Updates the state and zeta values based on the control law and system dynamics.
        
        :param hz: A vector used in the control law.
        """
        # Control input u
        self.u = np.dot(self.K, hz * self.zeta)  # u = K * (hz * zeta)
        
        # Update dzeta (diagonal of hz' * Gamma * hz)
        self.dzeta = np.diag(np.dot(np.dot(hz.T, self.Gamma), hz))
        
        # State update: dx = A * x + B * u
        self.dx = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        # Time update
        self.t += self.dt
        
        # Update the state vector and zeta
        self.x += self.dx * self.dt
        self.zeta += self.dzeta * self.dt
