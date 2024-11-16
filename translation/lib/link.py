import numpy as np

class Link:
    def __init__(self, A, B, P, xIniti, di, alphai, rhoInitij, thetaInitij,
                 xInitj, dj, alphaj, rhoInitji, thetaInitji, dt):
        """
        Constructor for the Link class.
        
        :param A: State matrix (nxn)
        :param B: Input matrix (nxr)
        :param P: A constant matrix for calculating Gamma
        :param xIniti: Initial position vector of agent i
        :param di: Some parameter for agent i
        :param alphai: Parameter for agent i
        :param rhoInitij: Initial rho for i-j link
        :param thetaInitij: Initial theta for i-j link
        :param xInitj: Initial position vector of agent j
        :param dj: Some parameter for agent j
        :param alphaj: Parameter for agent j
        :param rhoInitji: Initial rho for j-i link
        :param thetaInitji: Initial theta for j-i link
        :param dt: Time step
        """
        self.A = A
        self.B = B
        self.Gamma = np.dot(np.dot(P, np.dot(B, B.T)), P)  # Gamma = P * B * B' * P

        self.di = di
        self.dj = dj
        self.alphai = alphai
        self.alphaj = alphaj
        
        self.zij = xIniti - xInitj
        self.zji = -self.zij
        
        self.hzij = self.zij
        self.hzji = self.zji
        self.dhzij = np.zeros(A.shape[0])
        self.dhzji = np.zeros(A.shape[0])
        
        self.eij = np.zeros(A.shape[0])
        self.eji = np.zeros(A.shape[0])
        self.deij = np.zeros(A.shape[0])
        self.deji = np.zeros(A.shape[0])
        
        self.lambdaG = np.max(np.linalg.eigvals(self.Gamma))  # Max eigenvalue of Gamma
        self.lambdaP = np.min(np.linalg.eigvals(P))  # Min eigenvalue of P
        
        self.Phiij = 0
        self.Phiji = 0
        
        self.indicator = 0  # Indicator flag
        
        self.thetaij = thetaInitij
        self.thetaji = thetaInitji
        self.dthetaij = 0.0
        self.dthetaji = 0.0
        
        self.rhoij = rhoInitij
        self.rhoji = rhoInitji
        self.drhoij = 0.0
        self.drhoji = 0.0
        self.rhoij0 = rhoInitij
        self.rhoji0 = rhoInitji
        
        self.dt = dt
        self.t = 0

    def get_hz(self):
        """Returns hzij and hzji."""
        return self.hzij, self.hzji

    def get_zij(self):
        """Returns zij."""
        return self.zij

    def get_indicator(self):
        """Returns the indicator value."""
        return self.indicator

    def get_Phiij(self):
        """Returns Phiij."""
        return self.Phiij

    def get_Phiji(self):
        """Returns Phiji."""
        return self.Phiji

    def get_rhoij(self):
        """Returns rhoij."""
        return self.rhoij

    def get_rhoji(self):
        """Returns rhoji."""
        return self.rhoji

    def get_thetaij(self):
        """Returns thetaij."""
        return self.thetaij

    def get_thetaji(self):
        """Returns thetaji."""
        return self.thetaji

    def is_triggered(self, xi, ui, zetaij, xj, uj, zetaji):
        """
        Checks if the link is triggered and updates its state based on the condition.
        
        :param xi: Position vector of agent i
        :param ui: Input vector for agent i
        :param zetaij: Some state vector for the link
        :param xj: Position vector of agent j
        :param uj: Input vector for agent j
        :param zetaji: Some state vector for the link
        """
        self.zij = xi - xj
        self.zji = xj - xi
        
        # Update state based on inputs
        self.update_state(ui, zetaij, uj, zetaji)
        
        # Update Phi and Psi
        self.Phiij = self.triggering_function(self.alphai, self.di, zetaij,
                                              self.hzij, self.Gamma, self.eij, self.thetaij)
        self.Phiji = self.triggering_function(self.alphaj, self.dj, zetaji,
                                               self.hzji, self.Gamma, self.eji, self.thetaji)
        
        # Trigger check
        if (self.Phiij <= 0 and self.rhoij < 0.0) or (self.Phiji <= 0 and self.rhoji < 0.0):
            # Without Sleeping Mechanism or without Triggering Mechanism
            self.hzij = self.zij
            self.hzji = self.zji
            
            self.eij = np.zeros_like(self.eij)
            self.eji = np.zeros_like(self.eji)
            
            self.rhoij = self.rhoij0
            self.rhoji = self.rhoji0
            
            self.indicator = 1
        else:
            self.indicator = 0

    def update_state(self, ui, zetaij, uj, zetaji):
        """Updates the state of the link based on inputs and time step."""
        self.t += self.dt
        
        # Evaluate equations of motion
        self.eval_EOM(ui, zetaij, uj, zetaji)
        
        # Update states
        self.hzij += self.dhzij * self.dt
        self.hzji += self.dhzji * self.dt
        
        self.thetaij += self.dthetaij * self.dt
        self.thetaji += self.dthetaji * self.dt
        
        self.eij += self.deij * self.dt
        self.eji += self.deji * self.dt
        
        self.rhoij += self.drhoij * self.dt
        self.rhoji += self.drhoji * self.dt

    def eval_EOM(self, ui, zetaij, uj, zetaji):
        """Evaluates the equations of motion for the link."""
        self.dhzij = np.dot(self.A, self.hzij)
        self.dhzji = np.dot(self.A, self.hzji)
        
        # Update dtheta
        self.dthetaij = np.dot(self.eij.T, np.dot(self.Gamma, self.eij))
        self.dthetaji = np.dot(self.eji.T, np.dot(self.Gamma, self.eji))
        
        self.deij = np.dot(self.A, self.eij) + np.dot(self.B, ui)
        self.deji = np.dot(self.A, self.eji) + np.dot(self.B, uj)
        
        # Update rho using the sleeping mechanism
        self.drhoij = self.sleepping_mechanism(self.lambdaG, self.lambdaP,
                                                self.di, self.alphai, zetaij, self.rhoij, self.thetaij)
        self.drhoji = self.sleepping_mechanism(self.lambdaG, self.lambdaP,
                                                self.dj, self.alphaj, zetaji, self.rhoji, self.thetaji)
    
    def triggering_function(self, alpha, d, zeta, hz, Gamma, e, theta):
        """
        A placeholder for the triggering function.
        Replace with your actual implementation.
        """
        # Example of a simple triggering function
        return np.dot(e.T, np.dot(Gamma, e)) - alpha * (np.linalg.norm(zeta) ** 2) - d * np.linalg.norm(hz) - theta
    
    def sleepping_mechanism(self, lambdaG, lambdaP, d, alpha, zeta, rho, theta):
        """
        A placeholder for the sleeping mechanism function.
        Replace with your actual implementation.
        """
        return np.max([lambdaG - lambdaP, d * np.linalg.norm(zeta) - alpha * rho - theta])
