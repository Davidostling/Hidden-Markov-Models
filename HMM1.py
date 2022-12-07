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

T = int(matrixList[0][0]) #Number of A rows
N = int(matrixList[3][0]) #Number of Emissions                                                              

def initAlpha():
    firstAlpha = [0 for _ in range(T)]
    for i in range(T):
        firstAlpha[i] = B[i][O[0]] * pi[0][i]
    return firstAlpha

def forwardAlg():
    alpha = initAlpha()
    
    for i in range(1, N):
        tempAlpha = []
        for j in range(T):
            tempAlpha.append(sumCalc(alpha,j)*B[j][O[i]])
        alpha = tempAlpha

    print(sum(alpha))
    
def sumCalc(alpha, index):
    sumcalc = 0
    for i in range(T):
        sumcalc+= alpha[i] * A[i][index]
    return sumcalc

forwardAlg()
