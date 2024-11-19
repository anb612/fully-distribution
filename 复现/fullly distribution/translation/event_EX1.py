import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from event_AEP import guo2021event_AEP

# Initialize parameters
numK = 9
Aep = np.zeros(numK)
numSamples = 100
dt = 0.00001
tf = 3
w = 10
kTheta = 1
kRho = 1000
gamma = 0.0001

# Topology: Define the edges of the graph
s = np.array([1, 2, 2, 3]) - 1  # 0-indexed for Python
t = np.array([2, 3, 4, 4]) - 1  # 0-indexed for Python
G = nx.Graph()
G.add_edges_from(zip(s, t))
L = nx.laplacian_matrix(G).toarray()

# Number of agents (nodes) and edges
N = L.shape[0]  # 4 agents
M = len(s) * 2  # Number of edges (4 * 2 edges)

# Initialize the A matrix and B vector
A = np.array([[0, w], [-w, 0]])
B = np.array([[0], [1]])
n = A.shape[0]  # n = 2

# Main loop over different alpha values
for k in range(1, numK + 1):
    alpha = 0.1 * k
    aep = 0

    # Run the simulation for numSamples times
    for kk in range(numSamples):
        # Initialize random conditions
        xInit = 10 * np.random.rand(N * n)  # Initial states of the agents
        zetaInit = np.random.rand(M // 2)  # Initial zeta values
        thetaInit = kTheta * np.random.rand(M)  # Initial theta values
        rhoInit = kRho * np.random.rand(M)  # Initial rho values

        # Placeholder function for guo2021event_AEP (this needs to be implemented or defined elsewhere)
        aepk, zPlot12, zPlot23, zPlot24, zPlot34, xPlot1, xPlot2, xPlot3, xPlot4, tauPlot12, tauPlot21, indiPlot12, numData = guo2021event_AEP(
            alpha, dt, tf, s, t, A, B, xInit, zetaInit, thetaInit, rhoInit, gamma)

        # Accumulate AEP values
        aep += aepk

    # Average over the number of samples
    aep /= numSamples
    Aep[k - 1] = aep  # Store the AEP for each alpha

    print(f"Alpha {alpha}: AEP = {aep}")

# Visualization of the results
LW = 1.5  # Line width for plots

# Plot z12, z23, z24, z34
plt.figure(6)
plt.plot(dt * np.arange(numData), zPlot12[0, :], 'r.', label='z12 (1)', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot12[1, :], 'b.', label='z12 (2)', linewidth=LW)
plt.grid(True)
plt.legend()

plt.figure(7)
plt.plot(dt * np.arange(numData), zPlot23[0, :], 'r.', label='z23 (1)', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot23[1, :], 'b.', label='z23 (2)', linewidth=LW)
plt.grid(True)
plt.legend()

plt.figure(8)
plt.plot(dt * np.arange(numData), zPlot24[0, :], 'r.', label='z24 (1)', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot24[1, :], 'b.', label='z24 (2)', linewidth=LW)
plt.grid(True)
plt.legend()

plt.figure(9)
plt.plot(dt * np.arange(numData), zPlot34[0, :], 'r.', label='z34 (1)', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot34[1, :], 'b.', label='z34 (2)', linewidth=LW)
plt.grid(True)
plt.legend()

# Plot the agents' trajectories
plt.figure(22)
plt.plot(xPlot1[0, 0], xPlot1[1, 0], '*', color='g', linewidth=LW, label='agent 1')
plt.plot(xPlot2[0, 0], xPlot2[1, 0], '*', color='k', linewidth=LW, label='agent 2')
plt.plot(xPlot3[0, 0], xPlot3[1, 0], '*', color='b', linewidth=LW, label='agent 3')
plt.plot(xPlot4[0, 0], xPlot4[1, 0], '*', color='r', linewidth=LW, label='agent 4')
plt.plot(xPlot1[0, :], xPlot1[1, :], 'g', linewidth=LW)
plt.plot(xPlot2[0, :], xPlot2[1, :], 'k', linewidth=LW)
plt.plot(xPlot3[0, :], xPlot3[1, :], 'b', linewidth=LW)
plt.plot(xPlot4[0, :], xPlot4[1, :], 'r', linewidth=LW)
plt.grid(True)
plt.axis('equal')
plt.legend(loc='southoutside', orientation='horizontal')
plt.xlabel('$x^1_i(t)$, $i = 1 , \ldots, 4$', fontsize=12)
plt.ylabel('$x^2_i(t)$, $i = 1 , \ldots, 4$', fontsize=12)

# Plot event internal time
i = 0
count = np.zeros(numData)
lt = 0
cur_t = 0
plt.figure(23)

for k in range(2 * numData // 3):
    if indiPlot12[k] == 1:
        count[i] = (k - lt) * dt
        plt.plot([cur_t, cur_t], [0, count[i]], color='b', linewidth=2)
        plt.plot(cur_t, count[i], 'o', color='b', linewidth=2)
        cur_t += count[i]
        i += 1
        lt = k

plt.xlabel('Time $t$', fontsize=12)
plt.ylabel('Event interval', fontsize=12)
plt.grid(True)

# Plot a horizontal line at the minimum event interval
mtau = min(tauPlot12[numData - 1], tauPlot21[numData - 1])
plt.axhline(y=mtau, color='r', linestyle='--', linewidth=2)

plt.show()
