import numpy as np
class Agent:
    def __init__(self,  A, B, P, dt, initial_position, zeta_init):
        # self.index = index
        self.neighbors_num = 0
        self.neighbors_edges = [] # 记录self与neighbor的边的index
        
        self.A = A
        self.B = B
        self.K = -B.T @ P
        self.Gamma = P * (B @ B.T) @ P
        
        self.t = 0
        self.dt = dt
        
        self.x = initial_position
        self.dx = np.zeros((A.shape[0], 1))
        
        self.ut = np.zeros((B.shape[1], 1))
        
        self.zeta = zeta_init
        self.dzeta = 0
    
    
    @classmethod
    def get_position(agent):
         return agent.position
    
    @classmethod
    def get_zeta(agent):
        return agent.zeta

    @classmethod
    def update_state(agent, hz):
        agent.ut = agent.K @ (hz @ agent.zeta)
        agent.dzeta = np.diag(hz.T @ agent.Gamma * hz)  #np.diag() 返回矩阵的对角线元素
        agent.dx = agent.A @ agent.x + agent.B @ agent.u    # dx = Ax + Bu
        agent.t = agent.t + agent.dt
        
        agent.x += agent.dx * agent.dt  # x = x + dx 这边的*是逐元素相乘
        agent.zeta += agent.dzeta * agent.dt

    
    # def update_position(self, A, B, u_t, delta_t):
    #     x_d = A @ self.position + B @ u_t
    #     self.position += delta_t * x_d 

    # def add_neighbor(self, neighbor_edge):
    #         if neighbor_edge not in self.neighbors_edges:
    #             self.neighbors_edges.append(neighbor_edge)
    #             self.neighbors_num += 1
    
    # def is_neighbor(self, agent):
    #     return agent in self.neighbors
    

