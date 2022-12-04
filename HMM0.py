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
            


def multiplyMatrix(X,Y):

    result = [[0 for _ in range(len(Y[0]))] for _ in range(len(X))]
    
    for xRow in range(len(X)):
        for yColumn in range(len(Y[0])):
            for yRow in range(len(Y)):
                result[xRow][yColumn] += float(X[xRow][yRow]) * float(Y[yRow][yColumn])
    return result


def matrixToString(matrix):
    strMatrix = str(len(matrix))+ ' ' + str(len(matrix[0]))
    strMatrix += ' ' +' '.join(map(str, [el for row in matrix for el in row]))
    return strMatrix


A = initializeMatrix(int(matrixList[0][0]), int(matrixList[0][1]), matrixList[0][2:]) 
B = initializeMatrix(int(matrixList[1][0]), int(matrixList[1][1]), matrixList[1][2:])
pi = initializeMatrix(int(matrixList[2][0]), int(matrixList[2][1]), matrixList[2][2:]) 


state1 = multiplyMatrix(pi, A)
state2 = multiplyMatrix(state1, B)
print((matrixToString(state2)))


