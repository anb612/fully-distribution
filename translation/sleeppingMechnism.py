def sleeppingMechnism(lambdaG, lambdaP, di, alphai, zetaij, rhoij, thetaij):
    if rhoij > 0:
        drhoij = - lambdaG / lambdaP * (
            (di / alphai + di * (di - 1)) * zetaij * rhoij**2 +
            (2 * di / alphai * zetaij + 2) * rhoij +
            di / alphai * zetaij + 2 * thetaij
        )
    else:
        drhoij = 0
    return drhoij
