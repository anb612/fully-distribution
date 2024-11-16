import numpy as np
import random
from agent import Agent
class Edge:
    def __init__(self, A, B, P, di, dj, zij, alphai, alphaj, thetaInitij, thetaInitji, rhoInitij, rhoInitji, dt):
        # self.index = index
        # self.n = n
        # self.head = head
        # self.tail = tail
        # self.z_ij = np.zeros((n, 1))
        # self.z_ji = -self.z_ij
        # self.zeta_ij = np.random.random() # ζ_ij
        # self.zeta_ji = self.zeta_ij
        # self.e_ij = np.zeros((n, 1))
        # self.e_ji = np.zeros((n, 1))
        # self.theta_ij = np.random.random() # θ_ij
        # self.theta_ji = np.random.random()
        # self.rho_ij_bar = np.random.random() * 100 # ρ_bar
        # self.rho_ji_bar = np.random.random() * 100 # ρ_bar
        
        self.A = A
        self.B = B
        self.Gamma = P * (B @ B.T) @ P

        self.di = di
        self.dj = dj

        self.alphai = alphai
        self.alphaj = alphaj
        
        self.zij = zij
        self.zji = -self.zij
        
        self.hzij = self.zij
        self.hzji = self.zji
        self.dhzij = np.zeros(A.shape[0], 1)
        self.dhzji = np.zeros(A.shape[0], 1)
        
        self.eij = np.zeros(A.shape[0], 1)
        self.eji = np.zeros(A.shape[0], 1)
        self.deij = np.zeros(A.shape[0], 1)
        self.deji = np.zeros(A.shape[0], 1)
        
        self.lambdaG = max(np.linalg.eigvals(self.Gamma))
        self.lambdaP = min(np.linalg.eigvals(P))
        # Φ
        self.Phiij = 0
        self.Phiji = 0
        
        # self.indicator = 1
        self.indicator = 0
        
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
        return [self.hzij, self.hzji]
    

    def get_zij(self):
          return self.zij
    
    def get_indicator(self):
        return self.indicator
        
    def get_Phiij(self):
        return self.Phiij
    
    def get_Phiji(self):
        return self.Phiji
    
    def get_Rhoij(self):
        return self.rhoij
    
    def get_Rhoji(self):
        return self.rhoji
    
        
    def get_thetaij(self):
        return self.thetaij
    
    def get_thetaji(self):
        return self.thetaji
    
        
    def isTriggered(self, agent_i, xi, xj, ui, zetaij):
        self.zij = xi - xj  
        self.zji = xj - xi 

        Agent.update_state(agent_i, hz)       
        function obj = isTriggered(obj , xi , ui , zetaij , ...
                xj , uj , zetaji)
            # obj.zij = xi - xj;
            # obj.zji = xj - xi;
            
            updateState(obj , ui , zetaij , uj , zetaji);
            
            obj.Phiij = triggeringFunction(obj.alphai , obj.di , zetaij , ...
                obj.hzij , obj.Gamma , obj.eij , obj.thetaij); 
            
            obj.Phiji = triggeringFunction(obj.alphaj , obj.dj , zetaji , ...
                obj.hzji , obj.Gamma , obj.eji , obj.thetaji);
            
            if or(and(obj.Phiij <= 0 , obj.rhoij < 0.0) , ...
                    and(obj.Phiji <= 0 , obj.rhoji < 0.0))
#             % Without Sleeping Mechanism
# %             if or(obj.Phiij <= 0 , obj.Phiji <= 0)
#             % Without Triggering Mechanism
# %             if or(obj.rhoij < 0.0 , obj.rhoji < 0.0)
# %             e = 0.001;
# %             if or(and(obj.Phiij < e , obj.rhoij <= 0) , ...
# %                     and(obj.Phiji < e , obj.rhoji <= 0))
                
                obj.hzij = obj.zij;
                obj.hzji = obj.zji;
                
                obj.eij = zeros(size(obj.A , 1), 1);
                obj.eji = zeros(size(obj.A , 1), 1);
                
                obj.rhoij = obj.rhoij0;
                obj.rhoji = obj.rhoji0;
                
                obj.indicator = 1;
            else
                obj.indicator = 0;
            end
        end
        
        function obj = updateState(obj , ui , zetaij , uj , zetaji)
            obj.t = obj.t + obj.dt;
            
            evalEOM(obj , ui , zetaij , uj , zetaji);
            
            obj.hzij = obj.hzij + obj.dhzij .* obj.dt;
            obj.hzji = obj.hzji + obj.dhzji .* obj.dt;
            
            obj.thetaij = obj.thetaij + obj.dthetaij .* obj.dt;
            obj.thetaji = obj.thetaji + obj.dthetaji .* obj.dt;
            
            obj.eij = obj.eij + obj.deij .* obj.dt;
            obj.eji = obj.eji + obj.deji .* obj.dt;
            
            obj.rhoij = obj.rhoij + obj.drhoij * obj.dt;
            obj.rhoji = obj.rhoji + obj.drhoji * obj.dt;
        end
        
        function obj = evalEOM(obj , ui , zetaij , uj , zetaji)
            obj.dhzij = obj.A * obj.hzij;
            obj.dhzji = obj.A * obj.hzji;
            
            % Update dtheta
            obj.dthetaij = obj.eij' * obj.Gamma * obj.eij;
            obj.dthetaji = obj.eji' * obj.Gamma * obj.eji;
            
            obj.deij = obj.A * obj.eij + obj.B * ui;
            obj.deji = obj.A * obj.eji + obj.B * uj;
            
            obj.drhoij = sleeppingMechnism(obj.lambdaG , obj.lambdaP , ...
                obj.di , obj.alphai , zetaij , obj.rhoij , obj.thetaij);
            
            obj.drhoji = sleeppingMechnism(obj.lambdaG , obj.lambdaP , ...
                obj.dj , obj.alphaj , zetaji , obj.rhoji , obj.thetaji);
        end 


    
    
    # def update_data(self, head, tail, z_ij, zeta_ij, zeta_ji, e_ij, e_ji, theta_ij, theta_ji):
    #     self.head = head
    #     self.tail = tail
    #     self.z_ij = z_ij
    #     self.z_ji = -z_ij
    #     self.zeta_ij = zeta_ij
    #     self.zeta_ji = zeta_ji
    #     self.e_ij = e_ij
    #     self.e_ji = e_ji
    #     self.theta_ij = theta_ij
    #     self.theta_ji = theta_ji

    # def get_z_ij(self):
    #     return self.head.position - self.tail.position

    # def check_z_ij_norm(self, epsilon):
    #     return np.linalg.norm(self.z_ij, ord='fro') <= epsilon

    # def get_e(self, A, B, delta_t, u_i, u_j):
    #     e_ij_d = A @ self.e_ij + B @ u_i
    #     e_ji_d = A @ self.e_ji + B @ u_j
    #     e_ij_now =  self.e_ij + delta_t * e_ij_d
    #     e_ji_now =  self.e_ji + delta_t * e_ji_d
    #     return e_ij_now, e_ji_now
    
    # # 计算θ
    # def get_theta(self, e_ij, e_ji, gamma, delta_t):
    #     theta_ij_d = e_ij.T @ gamma @ e_ij
    #     theta_ji_d = e_ji.T @ gamma @ e_ji
    #     theta_ij_now = self.theta_ij + delta_t * theta_ij_d
    #     theta_ji_now = self.theta_ji + delta_t * theta_ji_d
    #     return theta_ij_now, theta_ji_now

    # # 计算ζ
    # def get_zeta(self, gamma, delta_t):
    #     z_ij = self.get_z_ij()
    #     z_ji = -z_ij
    #     zeta_ij_d = z_ij.T @ gamma @ z_ij
    #     zeta_ji_d = z_ji.T @ gamma @ z_ji
    #     zeta_ij_now = self.zeta_ij + delta_t * zeta_ij_d
    #     zeta_ji_now = self.zeta_ji + delta_t * zeta_ji_d
    #     return zeta_ij_now, zeta_ji_now
            

    # # 计算Φ_i
    # def get_phi(self, gamma, zeta_ij, zeta_ji, z_ij, z_ji, e_ij, e_ji, alpha_i = 0.5, alpha_j = 0.5):
    #     d_i = self.head.neighbors_num
    #     d_j = self.tail.neighbors_num
    #     phi_ij = (0.5 * alpha_i / d_i) @ self.zeta_ij @ self.theta_ij.T @ gamma @ self.theta_ij + self.zeta_ij @ self.theta_ij.T @ gamma @ self.e_ij - self.theta_ij @ self.e_ij.T @ gamma @ self.e_ij
    #     phi_ji = (0.5 * alpha_j / d_j) @ self.zeta_ji @ self.theta_ji.T @ gamma @ self.theta_ji + self.zeta_ji @ self.theta_ji.T @ gamma @ self.e_ji - self.theta_ji @ self.e_ji.T @ gamma @ self.e_ji
    #     return phi_ij, phi_ji
