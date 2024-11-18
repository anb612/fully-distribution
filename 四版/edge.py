import numpy as np
import random
import math
class Edge:
    def __init__(self, index, A, B, P, head, tail, init_zij, dt):
        self.index = index
        self.A = A
        self.B = B
        self.Gamma = P @ B @ B.T @ P  # Gamma = P * B * B' * P
        self.head = head
        self.tail = tail
        
        self.z_ij = init_zij
        self.z_ji = -self.z_ij
        self.hz_ij = init_zij
        self.hz_ji = -self.hz_ij
        
        self.zeta_ij = np.random.random() # ζ_ij
        self.zeta_ji = self.zeta_ij
        self.e_ij = np.zeros((A.shape[0], 1))
        self.e_ji = np.zeros((A.shape[0], 1))
        self.theta_ij = np.random.random() # θ_ij
        self.theta_ji = np.random.random()
        self.rho_ij_bar =  np.random.uniform(0, 100) # ρ_bar
        self.rho_ji_bar =  np.random.uniform(0, 100) # ρ_bar
        self.rho_ij = self.rho_ij_bar
        self.rho_ji = self.rho_ji_bar
        
        self.alpha_i = 0.5
        self.alpha_j = 0.5

        self.dt = dt
        
        self.Gamma_max = np.max(np.linalg.eigvals(self.Gamma))  # Max eigenvalue of Gamma
        self.P_min = np.min(np.linalg.eigvals(P))  # Min eigenvalue of P

    def update_hz(self):
        # 更新估计误差
        hz_ij_d = self.A @ self.hz_ij
        
        self.hz_ij += self.dt * hz_ij_d
        self.hz_ji = -self.hz_ij
        
    
    def update_zeta(self):
        # 更新ζ
        self.zeta_ij = self.hz_ij.T @ self.Gamma @ self.hz_ij
        self.zeta_ji = self.hz_ji.T @ self.Gamma @ self.hz_ji
        # zeta_ij_d = self.hz_ij.T @ self.Gamma @ self.hz_ij
        # zeta_ji_d = self.hz_ji.T @ self.Gamma @ self.hz_ji
        # self.zeta_ij += self.dt * zeta_ij_d
        # self.zeta_ji += self.dt * zeta_ji_d

    def update_e(self):
        # 更新e
        e_ij_d = self.A @ self.e_ij + self.B @ self.head.ut
        e_ji_d = self.A @ self.e_ji + self.B @ self.tail.ut
        self.e_ij += self.dt * e_ij_d
        self.e_ji += self.dt * e_ji_d     

    def update_theta(self):
        # 更新θ
        theta_ij_d = self.e_ij.T @ self.Gamma @ self.e_ij
        theta_ji_d = self.e_ji.T @ self.Gamma @ self.e_ji
        self.theta_ij += self.dt * theta_ij_d
        self.theta_ji += self.dt * theta_ji_d
    
    def update_rho(self):
        rho_ij_decay_rate, rho_ji_decay_rate = self.sleepping_mechnism()
        self.rho_ij += self.dt * rho_ij_decay_rate
        self.rho_ji += self.dt * rho_ji_decay_rate
    def trigger_function(self):
        d_i = self.head.neighbors_num
        d_j = self.tail.neighbors_num
        phi_ij = (0.5 * self.alpha_i / d_i) * self.zeta_ij @ self.hz_ij.T @ self.Gamma @ self.hz_ij + self.zeta_ij @ self.hz_ij.T @ self.Gamma @ self.e_ij - self.theta_ij @ self.e_ij.T @ self.Gamma @ self.e_ij
        phi_ji = (0.5 * self.alpha_j / d_j) * self.zeta_ji @ self.hz_ji.T @ self.Gamma @ self.hz_ji + self.zeta_ji @ self.hz_ji.T @ self.Gamma @ self.e_ji - self.theta_ji @ self.e_ji.T @ self.Gamma @ self.e_ji
        if phi_ij <= 0 or phi_ji <= 0 :
            return True
        else:
            return False
    
    def sleepping_mechnism(self):
        # 定义 rho 的演化函数
        d_i = self.head.neighbors_num
        d_j = self.tail.neighbors_num
        zeta_i_bar = self.head.zeta_max
        zeta_j_bar = self.tail.zeta_max
        if self.rho_ij <= 0:
            rho_ij_decay_rate = 0
        else:
            sign_rho_ij = -np.sign(self.rho_ij_bar) * (self.Gamma_max / self.P_min)
            term_1 =  ((d_i / self.alpha_i) * self.zeta_ij + d_i * (d_i - 1) * zeta_i_bar) * self.rho_ij_bar**2
            term_2 = 2 * ((d_i / self.alpha_i) * self.zeta_ij + 1) * self.rho_ij_bar + (d_i / self.alpha_i) * self.zeta_ij + 2 * self.theta_ij
            rho_ij_decay_rate = sign_rho_ij * (term_1 + term_2)
        
        if self.rho_ji <= 0:
            rho_ji_decay_rate = 0
        else:
            sign_rho_ji = -np.sign(self.rho_ji_bar) * (self.Gamma_max / self.P_min)
            term_1 =  ((d_j / self.alpha_j) * self.zeta_ji + d_j * (d_j - 1) * zeta_j_bar) * self.rho_ji_bar**2
            term_2 = 2 * ((d_j / self.alpha_j) * self.zeta_ji + 1) * self.rho_ji_bar + (d_j / self.alpha_j) * self.zeta_ji + 2 * self.theta_ji
            rho_ji_decay_rate = sign_rho_ji * (term_1 + term_2)

        return rho_ij_decay_rate, rho_ji_decay_rate
    
    
    def reset_e(self):
        self.e_ij = np.zeros((self.A.shape[0], 1))
        self.e_ji = np.zeros((self.A.shape[0], 1))

    
    def update_z(self, i_position, j_position):
        self.z_ij = i_position - j_position
        self.z_ji = -self.z_ij
        self.hz_ij = i_position - j_position
        self.hz_ji = -self.hz_ij

    def reset_rho(self):
        self.rho_ij = self.rho_ij_bar
        self.rho_ji = self.rho_ji_bar
    def calculate_sleeping_time(self):
        # 定义求睡眠时间的函数
        di = self.head.neighbors_num
        dj = self.tail.neighbors_num
        a = -self.Gamma_max / self.P_min * (di / self.alpha_i + di * (di - 1)) * self.zeta_ij 
        b = -self.Gamma_max / self.P_min * (2 * di / self.alpha_i * self.zeta_ij + 2)
        c = -self.Gamma_max / self.P_min * (di / self.alpha_i * self.zeta_ij + self.theta_ij)
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            ij_sleep_time = 2 / math.sqrt(-delta) * (math.atan(b / math.sqrt(-delta)) - math.atan((2 * a * self.rho_ij_bar + b) / math.sqrt(-delta)))
        elif delta == 0:
            ij_sleep_time = - 2 / b + 2 / (2 * a * self.rho_ij_bar + b)
        else:
            ij_sleep_time = 1 / math.sqrt(delta) * math.log((b - delta) * (2 * a * self.rho_ij_bar + b + math.sqrt(delta)) / ((b + math.sqrt(delta)) * (2 * a * self.rho_ij_bar + b - math.sqrt(delta))))
        
        a = -self.Gamma_max / self.P_min * (di / self.alpha_j + dj * (dj - 1)) * self.zeta_ji
        b = -self.Gamma_max / self.P_min * (2 * dj / self.alpha_j * self.zeta_ji + 2)
        c = -self.Gamma_max / self.P_min * (dj / self.alpha_j * self.zeta_ji + self.theta_ji)
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            ji_sleep_time = 2 / math.sqrt(-delta) * (math.atan(b / math.sqrt(-delta)) - math.atan((2 * a * self.rho_ij_bar + b) / math.sqrt(-delta)))
        elif delta == 0:
            ji_sleep_time = - 2 / b + 2 / (2 * a * self.rho_ij_bar + b)
        else:
            ji_sleep_time = 1 / math.sqrt(delta) * math.log((b - delta) * (2 * a * self.rho_ij_bar + b + math.sqrt(delta)) / ((b + math.sqrt(delta)) * (2 * a * self.rho_ij_bar + b - math.sqrt(delta))))
        
        # rho_ij_decay_rate, rho_ji_decay_rate = self.get_rho_decay_rate()
        # ij_sleep_time = self.rho_ij_bar / rho_ij_decay_rate
        # ji_sleep_time = self.rho_ji_bar / rho_ji_decay_rate
        return min(ij_sleep_time, ji_sleep_time)
    
    
    # def update_data(self, head, tail, z_ij, zeta_ij, zeta_ji, e_ij, e_ji, theta_ij, theta_ji):
    #     self.head = head
    #     self.tail = tail
    #     self.z_ij = z_ij
    #     self.z_ji = -self.z_ij
    #     self.zeta_ij = zeta_ij
    #     self.zeta_ji = self.zeta_ji
    #     self.e_ij = e_ij
    #     self.e_ji = e_ji
    #     self.theta_ij = theta_ij
    #     self.theta_ji = theta_ji




        
    # # 计算Φ_i
    # def get_phi(self, gamma, zeta_ij, zeta_ji, z_ij, z_ji, e_ij, e_ji, alpha_i = 0.5, alpha_j = 0.5):
    #     d_i = self.head.neighbors_num
    #     d_j = self.tail.neighbors_num
    #     phi_ij = (0.5 * alpha_i / d_i) @ self.zeta_ij @ self.theta_ij.T @ gamma @ self.theta_ij + self.zeta_ij @ self.theta_ij.T @ gamma @ self.e_ij - self.theta_ij @ self.e_ij.T @ gamma @ self.e_ij
    #     phi_ji = (0.5 * alpha_j / d_j) @ self.zeta_ji @ self.theta_ji.T @ gamma @ self.theta_ji + self.zeta_ji @ self.theta_ji.T @ gamma @ self.e_ji - self.theta_ji @ self.e_ji.T @ gamma @ self.e_ji
    #     return phi_ij, phi_ji
