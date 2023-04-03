# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:14:52 2023

Homework 3
Numerical Caculus
April 2, 2023

@author: Jacob Romero
"""

import numpy as np

#1 Euler Method with the following details
# a. Function: t – y^2
# b. Range: 0 < t < 2
# c. Iterations: 10
# d. Initial Point: f(0) = 1


def f(x,y):
    return x- (y*y)


def euler(x0,y0,xn,n):
    
    h = (xn-x0)/n
    
    for i in range(n):
        slope = f(x0, y0)
        ans = y0 + h * slope
        y0 = ans
        x0 = x0+h
    
    print(ans)
    print()

x0 = 0
xn = 2
y0 = 1
step = 10

euler(x0,y0,xn,step)

# 2 Runge-Kutta with the following details:
# a. Function: t – y^2
# b. Range: 0 < t < 2
# c. Iterations: 10
# d. Initial Point: f(0) = 1

def f(x,y):
    return x-(y*y)

def rk4(x0,y0,xn,n):
    
    h = (xn-x0)/n
    
    for i in range(n):
        k1 = h * (f(x0, y0))
        k2 = h * (f((x0+h/2), (y0+k1/2)))
        k3 = h * (f((x0+h/2), (y0+k2/2)))
        k4 = h * (f((x0+h), (y0+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        ans2 = y0 + k
        y0 = ans2
        x0 = x0+h
    
    print(ans2)
    print()
    
x0 = 0
xn = 2
y0 = 1
step = 10

rk4(x0,y0,xn,step)


#3 Use Gaussian elimination and backward substitution solve the following linear system of  
# equations written in augmented matrix format. 


# Define the augmented matrix
A = [[2, -1, 1, 6],
     [1, 3, 1, 0],
     [-1, 5, 4, -3]]
# Perform Gaussian elimination
for i in range(len(A)):
    # Find the row with the largest absolute value in the i-th column
    max_row = i
    for j in range(i+1, len(A)):
        if abs(A[j][i]) > abs(A[max_row][i]):
            max_row = j
    # Swap the current row with the row with the largest absolute value in the i-th column
    A[i], A[max_row] = A[max_row], A[i]
    # Reduce the i-th column to 1 by dividing the i-th row by A[i][i]
    pivot = A[i][i]
    for j in range(i, len(A[i])):
        A[i][j] /= pivot
    # Eliminate the i-th column in all other rows
    for j in range(len(A)):
        if j != i:
            factor = A[j][i]
            for k in range(i, len(A[i])):
                A[j][k] -= factor * A[i][k]
# Extract the solutions from the augmented matrix
x = [row[-1] for row in A]
# Print the solutions
print(x)
print()



#4 Implement LU Factorization for the following matrix and do the following:
# 1 1 0 3
# 2 1 −1 1
# 3 −1 −1 2
# −1 2 3 −1
# a. Print out the matrix determinant. 
# b. Print out the L matrix.
# c. Print out the U matrix

def lu_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]

    return L, U
def determinant(matrix):
    L, U = lu_decomposition(matrix)
    det = np.prod(np.diag(U))
    return det

A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

# Calculate the determinant, L matrix, and U matrix
det = determinant(A)
L, U = lu_decomposition(A)

# Print the results
print(det)
print()
print(L)
print()
print(U)
print()

#5 Determine if the following matrix is diagonally dominate.
# 9 0 5 2 1
# 3 9 1 2 1
# 0 1 7 2 3
# 4 2 3 12 2
# 3 2 4 0 8


A = [[9,0,5,2,1],
     [3,9,1,2,1],
     [4,2,3,12,2],
     [3,2,4,0,8]]

def isDiagonallyDominantMatrix(A):
    for i, row in enumerate(A):
        s = sum(abs(v) for j, v in enumerate(row) if i != j)
        if s > abs(row[i]):
            return False
    return True

print(isDiagonallyDominantMatrix(A))

print()

#6 Determine if the matrix is a positive definite.
# 2 2 1
# 2 3 0
# 1 0 2

# Creating numpy array
arr = np.array([[2,2,1],[2,3,0],[1,0,2]])

# Check all eigen value
res = np.all(np.linalg.eigvals(arr) > 0)

print(res)