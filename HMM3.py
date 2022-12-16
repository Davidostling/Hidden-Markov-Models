import sys
import math

matrixList = []
read = sys.stdin.read().split("\n")
for element in read:
    matrixList.append(element.split(" "))


def initializeMatrix(rows, columns, data):
    index = 0
    matrix = [[0 for _ in range(columns)] for _ in range(rows)]
    for row in range(rows):
        for column in range(columns):
            matrix[row][column] = float(data[index])
            index += 1

    return matrix


def multiplyMatrix(X, Y):

    result = [[0 for _ in range(len(Y[0]))] for _ in range(len(X))]

    for xRow in range(len(X)):
        for yColumn in range(len(Y[0])):
            for yRow in range(len(Y)):
                result[xRow][yColumn] += float(X[xRow]
                                               [yRow]) * float(Y[yRow][yColumn])
    return result


def matrixToStringPrint(matrix):
    strMatrix = str(len(matrix)) + ' ' + str(len(matrix[0]))
    strMatrix += ' ' + \
        ' '.join(map(str, [round(el, 6) for row in matrix for el in row]))
    print(strMatrix)


A = initializeMatrix(int(matrixList[0][0]), int(
    matrixList[0][1]), matrixList[0][2:])  # transition
B = initializeMatrix(int(matrixList[1][0]), int(
    matrixList[1][1]), matrixList[1][2:])  # emission matrix
pi = initializeMatrix(int(matrixList[2][0]), int(
    matrixList[2][1]), matrixList[2][2:])  # initial state prob
Ostr = matrixList[3][1:]
O = [int(i) for i in Ostr[:-1]]  # Sequence of Emissions

N = int(matrixList[0][0])  # Number of A rows

T = int(matrixList[3][0])  # Number of Emissions


def initAlpha():
    firstAlpha = [[0 for _ in range(N)] for _ in range(T)]
    c = [0]
    for i in range(N):
        firstAlpha[0][i] = B[i][O[0]] * pi[0][i]
        c[0] += firstAlpha[0][i]

    return firstAlpha, c


def sumAlphaCalc(alpha, index):
    sumcalc = 0
    for i in range(N):
        sumcalc += alpha[i] * A[i][index]
    return sumcalc


def forwardAlg():
    alpha, c = initAlpha()

    # timeStep
    c[0] = 1/c[0]
    for i in range(N):
        alpha[0][i] *= c[0]

    for i in range(1, T):
        tempAlpha = []
        c.append(0)
        for j in range(N):
            a = sumAlphaCalc(alpha[i-1], j)*B[j][O[i]]
            c[i] += a
            tempAlpha.append(a)

    # timeStep
        c[i] = 1 / c[i]
        for j in range(N):
            alpha[i][j] = tempAlpha[j] * c[i]

    return alpha, c


def initBeta(c):
    firstBeta = [[0 for _ in range(N)] for _ in range(T)]

    for i in range(N):
        firstBeta[-1][i] = c[-1]
    return firstBeta


def sumBetaCalc(beta, indexI, IndexJ):
    sumcalc = 0
    for i in range(N):
        sumcalc += beta[indexI+1][i] * A[IndexJ][i] * B[i][O[indexI+1]]
    return sumcalc


def backwardsAlg(c):
    beta = initBeta(c)

    for i in range(T-2, -1, -1):
        tempBeta = []
        for j in range(N):
            b = sumBetaCalc(beta, i, j)
            tempBeta.append(b)

        # timeStep
        for j in range(N):
            beta[i][j] = tempBeta[j] * c[i]

    return beta


def calcDiGamma(alpha, beta, indexI, indexJ, indexZ):
    currentDiGamma = alpha[indexI][indexJ] * A[indexJ][indexZ] * \
        B[indexZ][O[indexI+1]] * beta[indexI+1][indexZ]
    return currentDiGamma


def declareGammas():
    gamma = [[0 for _ in range(N)] for _ in range(T)]
    diGamma = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(T)]
    return gamma, diGamma


def handleEdgeCaseGamma(gamma, alpha):
    for i in range(N):
        gamma[T-1][i] = alpha[T-1][i]


def calcGamma(alpha, beta):
    gamma, diGamma = declareGammas()

    for i in range(T-1):
        for j in range(N):
            gamma[i][j] = 0  # Used to normalize gamma[i][j]
            for z in range(N):
                diGamma[i][j][z] = calcDiGamma(alpha, beta, i, j, z)
                gamma[i][j] = gamma[i][j] + diGamma[i][j][z]

    handleEdgeCaseGamma(gamma, alpha)

    return gamma, diGamma


# log(P(O|lambda))
def logProb():
    _, c = forwardAlg()
    logProb = map(math.log, c)
    return -sum(logProb)


def reEstimate():
    alpha, c = forwardAlg()
    beta = backwardsAlg(c)
    gamma, diGamma = calcGamma(alpha, beta)

    # PI Estimation
    for i in range(N):
        pi[0][i] = gamma[0][i]

    # A Estimation
    for i in range(N):
        denom = 0
        for j in range(T-1):
            denom += gamma[j][i]

        for j in range(N):
            nummer = 0
            for k in range(T-1):
                nummer += diGamma[k][i][j]
            A[i][j] = nummer/denom

    # B Estimation
    for i in range(N):
        denom = 0
        for j in range(T):
            denom += gamma[j][i]

        for j in range(len(B)):
            nummer = 0
            for k in range(T):
                if (int(O[k]) == j):
                    nummer += gamma[k][i]
            B[i][j] = nummer/denom


def baumWelch():
    oldLogProb = -math.inf
    i = 0

    while i < 100:
        reEstimate()
        log_prob = logProb()
        oldLogProb = min(oldLogProb, log_prob)

        if log_prob < oldLogProb:
            break
        i += 1

    matrixToStringPrint(A)
    matrixToStringPrint(B)


baumWelch()
