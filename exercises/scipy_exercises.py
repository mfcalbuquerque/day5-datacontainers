from scipy import linalg
import numpy as np


#a
print("\na)")
A = np.array([[1,-2,3],[4,5,6],[7,1,9]])
print("Matrix A: \n", A)

#b
print("\nb)")
b = np.array([1,2,3])
print("Vector b: \n", b)

#c
print("\nc)")
x = linalg.solve(A,b)
print("The solution to the linear equation is: \n", x)

#d
print("\nd)")
b_expected = A @ x
print("The expected solution (b = A x) is: \n", b_expected)

#e
print("\ne)")
B_other = np.array(np.random.randint(0, 21, size=9)).reshape((3,3))
x_other = linalg.solve(A,B_other)
print("B = ", B_other)
print("For B as a 3x3 matrix, x is: \n", x_other)
print("And B = A x is: \n", A @ x_other)

#f
print("\nf)")
w, vr = linalg.eig(A)
print("The eigenvalues are: \n", w)
print("The eigenvectors are: \n", vr)

#g
print("\ng)")
inv_det = linalg.det(linalg.inv(A))
print("The inverse, determinant of A is: \n", inv_det)

#h
print("\nh)")
for n in ['fro', 'nuc', 1, -1, 2, -2]:
    norm = linalg.norm(A, ord=n)
    print("norm(A) with order {} is:\n {}".format(n, norm))

