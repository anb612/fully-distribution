import numpy as np
import networkx as nx
from scipy.linalg import solve_riccati
from lib.agent import Agent
from lib.link import Link

def guo2021event_AEP(alpha, dt, tf, s, t, A, B, xInit, zetaInit, thetaInit, rhoInit, gamma):
    # This function is used to calculate the average event period.
    
    # Create graph from the given topology
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    
    # Calculate Laplacian matrix
    L = nx.laplacian_matrix(G).todense()
    N = L.shape[0]  # Number of agents
    M = len(s) * 2  # Number of edges (2 for each undirected edge)
    
    # Extract diagonal values of Laplacian
    d1, d2, d3, d4 = np.diag(L)

    # (5) Riccati Equation for P and Gamma matrices
    P = solve_riccati(A, 2 * B @ B.T, np.eye(A.shape[0]))
    Gamma = P @ (B @ B.T) @ P

    # Controller gain
    K = -B.T @ P
    
    lambdaG = np.max(np.linalg.eigvals(Gamma))
    lambdaP = np.min(np.linalg.eigvals(P))

    # Initialization
    n = A.shape[0]  # Dimension of state for each agent

    # Split the initial states
    xInit_1, xInit_2, xInit_3, xInit_4 = np.split(xInit, 4)
    
    zetaInit_12, zetaInit_23, zetaInit_24, zetaInit_34 = zetaInit
    zetaInit_1 = zetaInit_12
    zetaInit_2 = np.array([zetaInit_21, zetaInit_23, zetaInit_24])
    zetaInit_3 = np.array([zetaInit_32, zetaInit_43])
    zetaInit_4 = np.array([zetaInit_42, zetaInit_43])

    thetaInit_12, thetaInit_23, thetaInit_24, thetaInit_34 = thetaInit[:4]
    thetaInit_21, thetaInit_32, thetaInit_42, thetaInit_43 = thetaInit[4:]

    # Define the agents and links
    agent1 = Agent(A, B, P, dt, xInit_1, zetaInit_1)
    agent2 = Agent(A, B, P, dt, xInit_2, zetaInit_2)
    agent3 = Agent(A, B, P, dt, xInit_3, zetaInit_3)
    agent4 = Agent(A, B, P, dt, xInit_4, zetaInit_4)

    link12 = Link(A, B, P, agent1.x, d1, alpha[0], rhoInit_12, thetaInit_12, agent2.x, d2, alpha[1], rhoInit_21, thetaInit_21, dt)
    link23 = Link(A, B, P, agent2.x, d2, alpha[1], rhoInit_23, thetaInit_23, agent3.x, d3, alpha[2], rhoInit_21, thetaInit_32, dt)
    link24 = Link(A, B, P, agent2.x, d2, alpha[1], rhoInit_24, thetaInit_24, agent4.x, d4, alpha[3], rhoInit_42, thetaInit_42, dt)
    link34 = Link(A, B, P, agent3.x, d3, alpha[2], rhoInit_34, thetaInit_34, agent4.x, d4, alpha[3], rhoInit_43, thetaInit_43, dt)

    numData = int(np.floor(tf / dt))
    
    # Initialize arrays to store results
    zPlot12 = np.zeros((A.shape[0], numData))
    zPlot23 = np.zeros((A.shape[0], numData))
    zPlot24 = np.zeros((A.shape[0], numData))
    zPlot34 = np.zeros((A.shape[0], numData))

    xPlot1 = np.zeros((A.shape[0], numData))
    xPlot2 = np.zeros((A.shape[0], numData))
    xPlot3 = np.zeros((A.shape[0], numData))
    xPlot4 = np.zeros((A.shape[0], numData))

    PhiPlot12 = np.zeros(numData)
    PhiPlot21 = np.zeros(numData)

    rhoPlot12 = np.zeros(numData)
    rhoPlot21 = np.zeros(numData)

    zetaPlot1 = np.zeros((d1, numData))
    zetaPlot2 = np.zeros((d2, numData))
    zetaPlot3 = np.zeros((d3, numData))
    zetaPlot4 = np.zeros((d4, numData))

    thetaPlot12 = np.zeros((2, numData))
    thetaPlot23 = np.zeros((2, numData))
    thetaPlot24 = np.zeros((2, numData))
    thetaPlot34 = np.zeros((2, numData))

    tauPlot12 = np.zeros(numData)
    tauPlot21 = np.zeros(numData)
    tauPlot23 = np.zeros(numData)
    tauPlot32 = np.zeros(numData)
    tauPlot24 = np.zeros(numData)
    tauPlot42 = np.zeros(numData)
    tauPlot34 = np.zeros(numData)
    tauPlot43 = np.zeros(numData)

    indiPlot12 = np.zeros(numData)
    indiPlot23 = np.zeros(numData)
    indiPlot24 = np.zeros(numData)
    indiPlot34 = np.zeros(numData)

    tk = 0
    nek = 0
    netotal = 0
    k = 0

    while tk == 0 and k <= numData:
        # Compute hz values
        hz1 = link12.hzij
        hz2 = np.concatenate([link12.hzji, link23.hzij, link24.hzij])
        hz3 = np.concatenate([link23.hzji, link34.hzij])
        hz4 = np.concatenate([link24.hzji, link34.hzji])
        
        # Update agent states
        agent1.updateState(hz1)
        agent2.updateState(hz2)
        agent3.updateState(hz3)
        agent4.updateState(hz4)

        # Triggering conditions
        link12.isTriggered(agent1.x, agent1.u, agent1.zeta, agent2.x, agent2.u, agent2.zeta[0])
        link23.isTriggered(agent2.x, agent2.u, agent2.zeta[1], agent3.x, agent3.u, agent3.zeta[0])
        link24.isTriggered(agent2.x, agent2.u, agent2.zeta[2], agent4.x, agent4.u, agent4.zeta[0])
        link34.isTriggered(agent3.x, agent3.u, agent3.zeta[1], agent4.x, agent4.u, agent4.zeta[1])
        
        # Store states and variables
        agent1_state = agent1.getState()
        agent2_state = agent2.getState()
        agent3_state = agent3.getState()
        agent4_state = agent4.getState()
        
        z12 = link12.getZij()
        z23 = link23.getZij()
        z24 = link24.getZij()
        z34 = link34.getZij()

        zPlot12[:, k] = z12
        zPlot23[:, k] = z23
        zPlot24[:, k] = z24
        zPlot34[:, k] = z34

        xPlot1[:, k] = agent1_state
        xPlot2[:, k] = agent2_state
        xPlot3[:, k] = agent3_state
        xPlot4[:, k] = agent4_state

        rhoPlot12[k] = link12.getRhoij()
        rhoPlot21[k] = link12.getRhoji()

        zetaPlot1[:, k] = agent1.getZeta()
        zetaPlot2[:, k] = agent2.getZeta()
        zetaPlot3[:, k] = agent3.getZeta()
        zetaPlot4[:, k] = agent4.getZeta()

        thetaPlot12[0, k] = link12.getThetaij()
        thetaPlot12[1, k] = link12.getThetaji()
        thetaPlot23[0, k] = link23.getThetaij()
        thetaPlot23[1, k] = link23.getThetaji()
        thetaPlot24[0, k] = link24.getThetaij()
        thetaPlot24[1, k] = link24.getThetaji()
        thetaPlot34[0, k] = link34.getThetaij()
        thetaPlot34[1, k] = link34.getThetaji()

        # Calculate inter-event times
        tauPlot12[k] = minIntereventTimes(lambdaG, lambdaP, d1, alpha[0], zetaPlot1[0, k], thetaPlot12[0, k], rhoInit_12)
        tauPlot21[k] = minIntereventTimes(lambdaG, lambdaP, d2, alpha[1], zetaPlot2[0, k], thetaPlot12[1, k], rhoInit_21)
        tauPlot23[k] = minIntereventTimes(lambdaG, lambdaP, d2, alpha[1], zetaPlot2[1, k], thetaPlot23[0, k], rhoInit_23)
        tauPlot32[k] = minIntereventTimes(lambdaG, lambdaP, d3, alpha[2], zetaPlot3[0, k], thetaPlot23[1, k], rhoInit_32)
        tauPlot24[k] = minIntereventTimes(lambdaG, lambdaP, d2, alpha[1], zetaPlot2[2, k], thetaPlot24[0, k], rhoInit_24)
        tauPlot42[k] = minIntereventTimes(lambdaG, lambdaP, d4, alpha[3], zetaPlot4[0, k], thetaPlot24[1, k], rhoInit_42)
        tauPlot34[k] = minIntereventTimes(lambdaG, lambdaP, d3, alpha[2], zetaPlot3[1, k], thetaPlot34[0, k], rhoInit_34)
        tauPlot43[k] = minIntereventTimes(lambdaG, lambdaP, d4, alpha[3], zetaPlot4[1, k], thetaPlot34[1, k], rhoInit_43)

        indiPlot12[k] = link12.getIndicator()
        indiPlot23[k] = link23.getIndicator()
        indiPlot24[k] = link24.getIndicator()
        indiPlot34[k] = link34.getIndicator()

        netotal += indiPlot12[k] + 2 * (indiPlot23[k] + indiPlot24[k] + indiPlot34[k])

        # Calculate the mean state of all agents
        X = np.column_stack([agent1.x, agent2.x, agent3.x, agent4.x])
        Mean = np.mean(X, axis=1)
        
        # Compute V1 for stopping criterion
        delta = X - Mean[:, np.newaxis]
        V1 = np.sum([delta[:, i].T @ P @ delta[:, i] for i in range(N)])

        if V1 < gamma:
            if tk == 0:
                tk = k * dt
                nek = netotal
        else:
            tk = 0

        if nek != 0:
            aep_tmp = tk * M / nek
        else:
            aep_tmp = 0

        k += 1

    return aep_tmp, zPlot12, zPlot23, zPlot24, zPlot34, xPlot1, xPlot2, xPlot3, xPlot4, tauPlot12, tauPlot21, indiPlot12, numData
