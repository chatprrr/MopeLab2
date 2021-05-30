from random import randint
import math
import numpy as np
from prettytable import PrettyTable

N_var = 8
Y_max = (30 - N_var) * 10
Y_min = (20 - N_var) * 10
X1_min = -30
X1_max = 0
X2_min = 10
X2_max = 60
N = 5
matrix = []

p_list = (0.99, 0.98, 0.95, 0.90)
rkr_table = {2: (1.73, 1.72, 1.71, 1.69),
             6: (2.16, 2.13, 2.10, 2.00),
             8: (2.43, 4.37, 2.27, 2.17),
             10: (2.62, 2.54, 2.41, 2.29),
             12: (2.75, 2.66, 2.52, 2.39),
             15: (2.9, 2.8, 2.64, 2.49),
             20: (3.08, 2.96, 2.78, 2.62)}


matrix = [[randint(Y_min, Y_max) for n in range(N)] for k in range(3)]
x_norm = [[-1, 1, -1], [-1, -1, 1]]
print("Дано: Y_max = {}  Y_min = {}  X1_min = {}  X1_max = {}  X2_min = {}  X2_max = {}".format(Y_max, Y_min, X1_min,
                                                                                                X1_max, X2_min, X2_max))
print("Матриця планування для m = {}".format(N))

#Перевірка однорідності дисперсії за критерієм Романовського

average_Y1 = sum(matrix[0][j] for j in range(N)) / N
average_Y2 = sum(matrix[1][j] for j in range(N)) / N
average_Y3 = sum(matrix[2][j] for j in range(N)) / N

D_Y1 = sum([(j - average_Y1) ** 2 for j in matrix[0]]) / N
D_Y2 = sum([(j - average_Y2) ** 2 for j in matrix[1]]) / N
D_Y3 = sum([(j - average_Y3) ** 2 for j in matrix[2]]) / N

main_deviation = math.sqrt((2 * (2 * N - 2)) / (N * (N - 4)))

Fuv_1 = D_Y1 / D_Y2
Fuv_2 = D_Y3 / D_Y1
Fuv_3 = D_Y3 / D_Y2

TETAuv_1 = ((N - 2) / N) * Fuv_1
TETAuv_2 = ((N - 2) / N) * Fuv_2
TETAuv_3 = ((N - 2) / N) * Fuv_3

Ruv_1 = abs(TETAuv_1 - 1) / main_deviation
Ruv_2 = abs(TETAuv_2 - 1) / main_deviation
Ruv_3 = abs(TETAuv_3 - 1) / main_deviation


m = min(rkr_table, key=lambda x: abs(x - N))
p = 0
for ruv in (Ruv_1, Ruv_2, Ruv_3):
    if ruv > rkr_table[m][0]:
        print(f'\n Дисперсія неоднорідна! Змінимо m={N} to m={N + 1}\n')
        N += 1
### for rkr in range(len(rkr_table[m])):
###     if ruv < rkr_table[m][rkr]:
###         p = rkr
### temp = rkr_table[m][p]
### p2 = p_list[p]
### item_table = temp

for i in range(3):
    matrix[i].append(randint(Y_min, Y_max))

mx1 = sum(x_norm[0]) / 3
mx2 = sum(x_norm[1]) / 3

my = (average_Y1 + average_Y2 + average_Y3) / 3

a1 = sum([i ** 2 for i in x_norm[0]]) / 3
a2 = sum(x_norm[0][i] * x_norm[1][i] for i in range(3)) / 3
a3 = sum([i ** 2 for i in x_norm[1]]) / 3
a11 = (x_norm[0][0] * average_Y1 + x_norm[0][1] * average_Y2 + x_norm[0][2] * average_Y3) / 3
a22 = (x_norm[1][0] * average_Y1 + x_norm[1][1] * average_Y2 + x_norm[1][2] * average_Y3) / 3

B0 = np.linalg.det(
    [[my, mx1, mx2], [a11, a1, a2], [a22, a2, a3]]) / (np.linalg.det([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]]))
B1 = np.linalg.det(
    [[1, my, mx2], [mx1, a11, a2], [mx2, a22, a3]]) / (np.linalg.det([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]]))
B2 = np.linalg.det(
    [[1, mx1, my], [mx1, a1, a11], [mx2, a2, a22]]) / (np.linalg.det([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]]))

delta_x1 = math.fabs(X1_max - X1_min) / 2
delta_x2 = math.fabs(X2_max - X2_min) / 2
x10 = (X1_max + X1_min) / 2
x20 = (X2_max + X2_min) / 2
a2_0 = B0 - (B1 * (x10 / delta_x1)) - (B2 * (x20 / delta_x2))
a2_1 = B1 / delta_x1
a2_2 = B2 / delta_x2


table_1 = PrettyTable()
table_1.add_column("X1", x_norm[0])
table_1.add_column("X2", x_norm[1])
table_1.add_column("Y1", [matrix[i][0] for i in range(3)])
table_1.add_column("Y2", [matrix[i][1] for i in range(3)])
table_1.add_column("Y3", [matrix[i][2] for i in range(3)])
table_1.add_column("Y4", [matrix[i][3] for i in range(3)])
table_1.add_column("Y5", [matrix[i][4] for i in range(3)])
print(table_1)

print("1) Перевірка однорідності дисперсії за критерієм Романовського:")
print("Однорідність дисперсій підтверджується з ймовірністю p = {} ".format(p2))
print("---------------------------------------------------------------")
print("2) Нормоване рівняння регресії : y = {} + ({})*x1 + ({})*x2 ".format("%.2f" % B0, "%.2f" % B1, "%.2f" % B2))
print("B0 - B1 - B2 = {} = Y1 = {}".format("%.2f" % (B0 - B1 - B2), average_Y1))
print("B0 + B1 - B2 = {} = Y2 = {}".format("%.2f" % (B0 + B1 - B2), average_Y2))
print("B0 - B1 + B2 = {} = Y3 = {}".format("%.2f" % (B0 - B1 + B2), average_Y3))
print("Результати збігається з середніми значеннями Yj !")
print("---------------------------------------------------------------")
print("3) Натур. коефіцієнтів")
print("Δx1 = {}  Δx2 = {}  X10 = {}  X20 = {}".format(delta_x1, delta_x2, x10, x20))
print("a0 = {}  a1 = {}  a2 = {}".format("%.2f" % a2_0, "%.2f" % a2_1, "%.2f" % a2_2))
print(
    "Натур. рівняння регресії: y = {} + ({})*x1 + ({})*x2 ".format("%.2f" % a2_0, "%.2f" % a2_1, "%.2f" % a2_2))
print("Перевірка по рядках:")
print("a2_0 + a2_1*X1_min + a2_2*X2_min = {} = Y1 = {}".format("%.2f" % (a2_0 + a2_1 * X1_min + a2_2 * X2_min),
                                                               average_Y1))
print("a2_0 + a2_1*X1_max + a2_2*X2_min = {} = Y2 = {}".format("%.2f" % (a2_0 + a2_1 * X1_max + a2_2 * X2_min),
                                                               average_Y2))
print("a2_0 + a2_1*X1_min + a2_2*X2_max = {} = Y3 = {}".format("%.2f" % (a2_0 + a2_1 * X1_min + a2_2 * X2_max),
                                                               average_Y3))
print("---------------------------------------------------------------")
print("Коефіцієнти натур. рівняння регресії правильні")
