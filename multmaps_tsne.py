import numpy as np
import matplotlib.pyplot as plt

def mult_maps_tsne(P, no_maps=5, no_dims=2, max_iter=200):
    eps = 1e-7

    # Normalize P to sum to one
    P /= np.sum(P)

    n = P.shape[0]  # Number of data points
    momentum = 0.5
    final_momentum = 0.8
    mom_switch_iter = 250
    epsilonY = 250  # Learning rate for Y
    epsilonW = 100  # Learning rate for weights

    # multiply P values by 4
    P *= 4

    # Initialize variables
    Y = np.random.randn(n, no_dims, no_maps) * 0.001
    y_incs = np.zeros((n, no_dims, no_maps))
    weights = np.full((n, no_maps), 1 / no_maps)

    num = np.zeros((n, n, no_maps))
    QQ = np.zeros((n, n, no_maps))
    dCdP = np.zeros((n, no_maps))
    dCdD = np.zeros((n, n, no_maps))
    dCdY = np.zeros((n, no_dims, no_maps))
    for iter in range(1, max_iter + 1):
        # Compute mixture proportions
        proportions = np.exp(-weights)
        proportions /= np.sum(proportions, axis=1, keepdims=True)

        # Compute pairwise affinities per map
        for m in range(no_maps):
            sum_Y = np.sum(Y[:, :, m] ** 2, axis=1)
            num[:, :, m] = 1 / (1 + sum_Y[:, None] + sum_Y[None, :] - 2 * Y[:, :, m] @ Y[:, :, m].T)
            np.fill_diagonal(num[:, :, m], 0)

        # Compute pairwise affinities under the mixture model
        QZ = np.full((n, n), eps)
        for m in range(no_maps):
            QQ[:, :, m] = (proportions[:, m][:, None] * proportions[:, m][None, :]) * num[:, :, m]
            QZ += QQ[:, :, m]

        Z = np.sum(QZ)
        Q = QZ / Z

        # Compute derivative of cost function w.r.t. mixture proportions
        PQ = Q - P
        tmp = PQ / QZ
        for m in range(no_maps):
            dCdP[:, m] = np.sum(proportions[:, m] * num[:, :, m] * tmp, axis=1)
        dCdP *= 2

        # Compute derivative of cost function w.r.t. mixture weights
        dCdW = proportions * (np.sum(dCdP * proportions, axis=1, keepdims=True) - dCdP)

        # Compute derivative of cost function w.r.t. pairwise distances
        for m in range(no_maps):
            dCdD[:, :, m] = (QQ[:, :, m] / QZ) * (-PQ) * num[:, :, m]

        # Compute derivative of cost function w.r.t. the maps
        for m in range(no_maps):
            for i in range(n):
                dCdY[i, :, m] = np.sum((dCdD[i, :, m][:, None] + dCdD[:, i, m][:, None]) * (Y[i, :, m] - Y[:, :, m]), axis=0)

        # Update the solution
        y_incs = momentum * y_incs - epsilonY * dCdY
        Y += y_incs
        Y -= np.mean(Y, axis=0)
        weights -= epsilonW * dCdW

        if iter == mom_switch_iter:
            momentum = final_momentum
        if iter == 50:
            P /= 4

        if iter % 25 == 0:
            C = np.sum(P * np.log(np.maximum(P, eps) / np.maximum(Q, eps)))
            print(f"Iteration {iter}: error is {C}")
              
    return Y, proportions

