import numpy as np
import random
class Edge:
    def __init__(self, index, n, m, head, tail):
        self.index = index
        self.n = n
        self.head = head
        self.tail = tail
        self.z_ij = np.zeros((n, 1))
        self.z_ji = -self.z_ij
        self.zeta_ij = np.random.random() # ζ_ij
        self.zeta_ji = self.zeta_ij
        self.e_ij = np.zeros((n, 1))
        self.e_ji = np.zeros((n, 1))
        self.theta_ij = np.random.random() # θ_ij
        self.theta_ji = np.random.random()
        self.rho_ij_bar = np.random.random() * 100 # ρ_bar
        self.rho_ji_bar = np.random.random() * 100 # ρ_bar

    def update_data(self, head, tail, z_ij, zeta_ij, zeta_ji, e_ij, e_ji, theta_ij, theta_ji):
        self.head = head
        self.tail = tail
        self.z_ij = z_ij
        self.z_ji = -z_ij
        self.zeta_ij = zeta_ij
        self.zeta_ji = zeta_ji
        self.e_ij = e_ij
        self.e_ji = e_ji
        self.theta_ij = theta_ij
        self.theta_ji = theta_ji

    def get_z_ij(self):
        return self.head.position - self.tail.position

    def check_z_ij_norm(self, epsilon):
        return np.linalg.norm(self.z_ij, ord='fro') <= epsilon

    def get_e(self, A, B, delta_t, u_i, u_j):
        e_ij_d = A @ self.e_ij + B @ u_i
        e_ji_d = A @ self.e_ji + B @ u_j
        e_ij_now =  self.e_ij + delta_t * e_ij_d
        e_ji_now =  self.e_ji + delta_t * e_ji_d
        return e_ij_now, e_ji_now
    
    # 计算θ
    def get_theta(self, e_ij, e_ji, gamma, delta_t):
        theta_ij_d = e_ij.T @ gamma @ e_ij
        theta_ji_d = e_ji.T @ gamma @ e_ji
        theta_ij_now = self.theta_ij + delta_t * theta_ij_d
        theta_ji_now = self.theta_ji + delta_t * theta_ji_d
        return theta_ij_now, theta_ji_now

    # 计算ζ
    def get_zeta(self, gamma, delta_t):
        z_ij = self.get_z_ij()
        z_ji = -z_ij
        zeta_ij_d = z_ij.T @ gamma @ z_ij
        zeta_ji_d = z_ji.T @ gamma @ z_ji
        zeta_ij_now = self.zeta_ij + delta_t * zeta_ij_d
        zeta_ji_now = self.zeta_ji + delta_t * zeta_ji_d
        return zeta_ij_now, zeta_ji_now
            

    # # 计算Φ_i
    # def get_phi(self, gamma, zeta_ij, zeta_ji, z_ij, z_ji, e_ij, e_ji, alpha_i = 0.5, alpha_j = 0.5):
    #     d_i = self.head.neighbors_num
    #     d_j = self.tail.neighbors_num
    #     phi_ij = (0.5 * alpha_i / d_i) @ self.zeta_ij @ self.theta_ij.T @ gamma @ self.theta_ij + self.zeta_ij @ self.theta_ij.T @ gamma @ self.e_ij - self.theta_ij @ self.e_ij.T @ gamma @ self.e_ij
    #     phi_ji = (0.5 * alpha_j / d_j) @ self.zeta_ji @ self.theta_ji.T @ gamma @ self.theta_ji + self.zeta_ji @ self.theta_ji.T @ gamma @ self.e_ji - self.theta_ji @ self.e_ji.T @ gamma @ self.e_ji
    #     return phi_ij, phi_ji
