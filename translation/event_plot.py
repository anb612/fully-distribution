import numpy as np
import matplotlib.pyplot as plt

# Define line width
LW = 1.5

# Define the data arrays (e.g., zPlot12, zPlot23, etc.)
# These should be defined elsewhere in your code, here they are placeholders
numData = 100  # Example length of data (replace with your actual data length)
dt = 0.0001  # Example time step (replace with your actual time step)
zPlot12 = np.random.rand(2, numData)  # Placeholder data for zPlot12
zPlot23 = np.random.rand(2, numData)  # Placeholder data for zPlot23
zPlot24 = np.random.rand(2, numData)  # Placeholder data for zPlot24
zPlot34 = np.random.rand(2, numData)  # Placeholder data for zPlot34

# Plot z12
plt.figure(6)
plt.plot(dt * np.arange(numData), zPlot12[0, :], 'r.', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot12[1, :], 'b.', linewidth=LW)
plt.grid(True)

# Plot z23
plt.figure(7)
plt.plot(dt * np.arange(numData), zPlot23[0, :], 'r.', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot23[1, :], 'b.', linewidth=LW)
plt.grid(True)

# Plot z24
plt.figure(8)
plt.plot(dt * np.arange(numData), zPlot24[0, :], 'r.', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot24[1, :], 'b.', linewidth=LW)
plt.grid(True)

# Plot z34
plt.figure(9)
plt.plot(dt * np.arange(numData), zPlot34[0, :], 'r.', linewidth=LW)
plt.plot(dt * np.arange(numData), zPlot34[1, :], 'b.', linewidth=LW)
plt.grid(True)

# For tauPlot12, tauPlot21 (assuming tauPlot12 and tauPlot21 are defined elsewhere)
tauPlot12 = np.random.rand(numData)  # Placeholder data for tauPlot12
tauPlot21 = np.random.rand(numData)  # Placeholder data for tauPlot21

# Plot tau12 tau21
plt.figure(18)
plt.plot(dt * np.arange(numData), tauPlot12, 'r.', linewidth=LW)
plt.plot(dt * np.arange(numData), tauPlot21, 'b.', linewidth=LW)
plt.grid(True)

# Define xPlot1, xPlot2, xPlot3, xPlot4 (example placeholders for agent positions)
xPlot1 = np.random.rand(2, numData)  # Placeholder data for agent 1
xPlot2 = np.random.rand(2, numData)  # Placeholder data for agent 2
xPlot3 = np.random.rand(2, numData)  # Placeholder data for agent 3
xPlot4 = np.random.rand(2, numData)  # Placeholder data for agent 4

# Plot agent trajectories
plt.figure(22)
plt.plot(xPlot1[0, 0], xPlot1[1, 0], '*', color='g', linewidth=LW)
plt.plot(xPlot2[0, 0], xPlot2[1, 0], '*', color='k', linewidth=LW)
plt.plot(xPlot3[0, 0], xPlot3[1, 0], '*', color='b', linewidth=LW)
plt.plot(xPlot4[0, 0], xPlot4[1, 0], '*', color='r', linewidth=LW)

plt.plot(xPlot1[0, :], xPlot1[1, :], 'g', linewidth=LW)
plt.plot(xPlot2[0, :], xPlot2[1, :], 'k', linewidth=LW)
plt.plot(xPlot3[0, :], xPlot3[1, :], 'b', linewidth=LW)
plt.plot(xPlot4[0, :], xPlot4[1, :], 'r', linewidth=LW)

plt.grid(True)
plt.axis('equal')

# Add legend
plt.legend(['agent 1', 'agent 2', 'agent 3', 'agent 4'], loc='southoutside', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=10)

# Add axis labels with LaTeX formatting
plt.xlabel(r'$x^1_i(t)$, $i = 1 , \ldots, 4$', fontsize=12)
plt.ylabel(r'$x^2_i(t)$, $i = 1 , \ldots, 4$', fontsize=12)

# Plot event internal time (assuming `indiPlot12` is defined elsewhere)
indiPlot12 = np.random.choice([0, 1], size=numData)  # Placeholder for event indicators
count = np.zeros(numData)
lt = 0
cur_t = 0

plt.figure(23)
i = 0
for k in range(2 * numData // 3):
    if indiPlot12[k] == 1:
        count[i] = (k - lt) * dt
        plt.plot([cur_t, cur_t], [0, count[i]], color='b', linewidth=2)
        plt.plot(cur_t, count[i], 'o', color='b', linewidth=2)
        cur_t = cur_t + count[i]
        i += 1
        lt = k

plt.xlabel(r'Time $t$', fontsize=12)
plt.ylabel(r'Event interval', fontsize=12)
plt.grid(True)

# Event threshold line
mtau = min(tauPlot12[-1], tauPlot21[-1])
plt.axhline(y=mtau, color='r', linestyle='--', linewidth=2)

plt.show()
