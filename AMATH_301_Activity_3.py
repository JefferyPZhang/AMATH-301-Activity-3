import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt


#n values
n = 10
n_list = [10]

# initialize A, b, b2
def A_init(n):
    A = np.zeros((n, n))
    for i in range (n):
        for j in range (n):
            if (i == j):
                A[i][j] = 2
                if (i - 1 >= 0):
                    A[i - 1][j] = -1
                if (i + 1 <= n - 1):
                    A[i + 1][j] = 1
    return A
                
def b_init(n): 
    b = np.zeros((n))
    for i in range (n):
        b[i] = i
    return b

def b2_init(n): 
    b2 = np.zeros((n))
    for i in range (n):
        b2[i] = n - i - 1
    return b2

A1 = A_init(n)
A2 = np.linalg.cond(A1)
A3 = b_init(n)

# using linalg.solve
start_time = time.time()
A4 = np.linalg.solve(A1, A3)
end_time = time.time()

# using LU decomp.
start_time = time.time()
P, L, U = scipy.linalg.lu(A1)
P = P.T
y = scipy.linalg.solve_triangular(L, P @ A3, lower = True)
A8 = scipy.linalg.solve_triangular(U, y)
end_time = time.time()
A5 = L
A6 = U
A7 = y

# using inverse
start_time = time.time()
A9 = np.linalg.inv(A1)
x = A9 @ A3
end_time = time.time()

# large matrices

solve_times = np.array([])
lu_times = np.array([])
inv_times = np.array([])

for n in n_list:
    print("n =", n)
    A = A_init(n)
    b = b_init(n)

    start_time = time.time()
    x = np.linalg.solve(A, b)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Solve time =", time_elapsed)
    solve_times = np.append(solve_times, time_elapsed)

    start_time = time.time()
    P, L, U = scipy.linalg.lu(A)
    P = P.T
    y = scipy.linalg.solve_triangular(L, P @ b, lower = True)
    x = scipy.linalg.solve_triangular(U, y)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("LU decomposition time =", time_elapsed)
    lu_times = np.append(lu_times, time_elapsed)

    start_time = time.time()
    A_inv = np.linalg.inv(A)
    x = A_inv @ b
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Inverse time =", time_elapsed, "\n")
    inv_times = np.append(inv_times, time_elapsed)

plt.plot(n_list, solve_times)
plt.plot(n_list, lu_times)
plt.plot(n_list, inv_times)
plt.xlabel("Size of System (num of rows and col.)")
plt.ylabel("Time to solve (in sec.)")
plt.title("Time to Solve Linear Systems with Different Methods")
plt.legend(("Solve", "LU", "Inverse"))
plt.show()

# large matrices + reverse
solve_times_2 = np.array([])
lu_times_2 = np.array([])

for n in n_list:
    print("n =", n)
    A = A_init(n)
    b = b_init(n)
    b2 = b2_init(n)

    start_time = time.time()
    x = np.linalg.solve(A, b)
    x2 = np.linalg.solve(A, b2)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Solve time =", time_elapsed)
    solve_times_2 = np.append(solve_times_2, time_elapsed)

    start_time = time.time()
    P, L, U = scipy.linalg.lu(A)
    P = P.T
    y2 = scipy.linalg.solve_triangular(L, P @ b2, lower = True)
    x = scipy.linalg.solve_triangular(U, y2)
    end_time = time.time()
    print("LU decomposition time =", time_elapsed, "\n")
    lu_times_2 = np.append(lu_times_2, time_elapsed)
    
    if n == 10:
        A10 = b2
        A11 = y2
        print(A11)
        A12 = x2

plt.plot(n_list, solve_times_2)
plt.plot(n_list, lu_times_2)
plt.xlabel("Size of System (num of rows and col.)")
plt.ylabel("Time to solve (in sec.)")
plt.title("Time to Solve Linear Systems with Different Methods")
plt.legend(("Solve", "LU"))
plt.show()