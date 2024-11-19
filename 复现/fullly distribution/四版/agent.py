import numpy as np
class Agent:
    def __init__(self, A, B, P, index, initial_position, dt):
        self.A = A
        self.B = B
        self.K = -B.T @ P
        self.Gamma = P @ (B @ B.T) @ P
        self.index = index
        self.position = initial_position
        self.ut = np.zeros((B.shape[1], 1))
        self.dt = dt
        self.zeta_max = -100000
        self.neighbors_num = 0
        self.neighbors_edges = [] # 记录self与neighbor的边的index

    def update_position(self):
        x_d = self.A @ self.position + self.B @ self.ut
        self.position += self.dt * x_d 
    
    def update_ut(self, item): 
        # item 是ut公式后面的加权和
        self.ut = self.K @ item
    
    def update_zeta_max(self, edges):
        if self.index == edges[self.neighbors_edges[0]].head.index:
            self.zeta_max =  edges[self.neighbors_edges[0]].zeta_ij
        else:
            self.zeta_max =  edges[self.neighbors_edges[0]].zeta_ji
            
        for neighbor_edge in self.neighbors_edges:
            if self.index == edges[neighbor_edge].head.index:
                self.zeta_max = max(self.zeta_max, edges[neighbor_edge].zeta_ij)
            else:
                self.zeta_max = max(self.zeta_max, edges[neighbor_edge].zeta_ji)
              
    def add_neighbor(self, neighbor_edge):
            if neighbor_edge not in self.neighbors_edges:
                self.neighbors_edges.append(neighbor_edge)
                self.neighbors_num += 1
    
    # def is_neighbor(self, agent):
    #     return agent in self.neighbors
    

