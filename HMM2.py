import sys

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
            index+=1
    
    return matrix

A = initializeMatrix(int(matrixList[0][0]), int(matrixList[0][1]), matrixList[0][2:]) #transition
B = initializeMatrix(int(matrixList[1][0]), int(matrixList[1][1]), matrixList[1][2:]) #emission matrix
pi = initializeMatrix(int(matrixList[2][0]), int(matrixList[2][1]), matrixList[2][2:]) #initial state prob  
Ostr =  matrixList[3][1:]                                                              
O = [int(i) for i in Ostr[:-1]]  # Sequence of Emissions                                                          

N = int(matrixList[0][0]) #Number of A rows
T = int(matrixList[3][0]) #Number of Emissions                                                              



def initDeltaMatrix(value, delta):
    firstDelta = [[value for _ in range(N)] for _ in range(T)]
    if(delta):
        for i in range(N):
            firstDelta[0][i] = B[i][O[0]] * pi[0][i]
    return firstDelta


def calcBestPath(matrix, indexI, indexJ):
    maxN = 0.0
    index = -1
    for k in range(N):
        currentValue =  A[k][indexJ] * B[indexJ][O[indexI]] * matrix[indexI-1][k]
        if(currentValue > maxN):
            maxN = currentValue
            index = k
    return maxN, index

def deltaAlg():
    deltaMatrix = initDeltaMatrix(0, True)
    deltaMatrixIndex =  initDeltaMatrix(-1, False)
  
    for i in range(1, T):
        for j in range(N):
            maxN, index = calcBestPath(deltaMatrix,i,j)
            deltaMatrix[i][j] = maxN
            deltaMatrixIndex[i-1][j]= index

    return deltaMatrix, deltaMatrixIndex

def calcMax():
    deltaMatrix, deltaMatrixIndex = deltaAlg()   
    result = []
    result.append(deltaMatrix[-1].index(max(deltaMatrix[-1]))) 
    j = 0
    for i in reversed(range(0, T-1)):
        path = deltaMatrixIndex[i][result[j]]
        result.append(path)
        j+=1
    
    result.reverse()
    return ' '.join(map(str, result))

p = calcMax()
print(p)
    










