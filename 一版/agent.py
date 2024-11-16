import numpy as np
class Agent:
    def __init__(self, n, m, index, initial_position):
        self.n = n
        self.m = m
        self.index = index
        self.position = initial_position
        # self.u_t = np.zeros((m, 1))
        self.neighbors_num = 0
        self.neighbors_edges = [] # 记录self与neighbor的边的index

    def update_position(self, A, B, u_t, delta_t):
        x_d = A @ self.position + B @ u_t
        self.position += delta_t * x_d 

    def add_neighbor(self, neighbor_edge):
            if neighbor_edge not in self.neighbors_edges:
                self.neighbors_edges.append(neighbor_edge)
                self.neighbors_num += 1
    
    # def is_neighbor(self, agent):
    #     return agent in self.neighbors
    

