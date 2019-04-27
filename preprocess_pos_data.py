import numpy as np
from aflow import *
import os
import Gen_atom
data_path = './data'
train_data = np.load(os.path.join(data_path,'train_data.npy'))
atom = Gen_atom.atom("Fe")
def lattice_constant(train_data, index):
    # Returns geometrical data describing the unit cell in the usual a,b,c,alpha,beta,gamma notation.
    material_i = train_data[index]
    a= material_i[3]
    b= material_i[4]
    c= material_i[5]
    retrun a, b, c


def find_min_nonzero(array):
    nonzero_array = array[np.nonzero(array)]
    min_value = nonzero_array[np.argmin(nonzero_array)]
    index = list(array).index(min_value)
    return index


def expand_cell(position_frac, a, b, c, n_a, n_b, n_c):
    N = position_frac.shape[0]  # number of atoms
    # expand the unit cell. n_a,n_b and n_c represent the number of repetitions along a,b,c axis.
    position_frac[:, 0] = (position_frac[:, 0] + (n_a - 1) * np.ones(N)) * a
    position_frac[:, 1] = (position_frac[:, 1] + (n_b - 1) * np.ones(N)) * b
    position_frac[:, 2] = (position_frac[:, 2] + (n_c - 1) * np.ones(N)) * c
    return position_frac


def find_min_dis(array_x, array_y):
    n_equivalent = array_x.shape[0]
    min_ = []
    for i in range(n_equivalent):
        array_x_i = np.ones((n_equivalent, 1)) * array_x[i]
        dis = np.sqrt(np.sum((array_x_i - array_y) ** 2, axis=1))
        min_dis = dis[find_min_nonzero(dis)]
        min_.append(min_dis)
    min = np.array(min_).min()
    return min


def get_dis_adj_matrix(position_frac, a, b, c):
    N = position_frac.shape[0]  # number of atoms
    # expand the unit cell into 2*2*2 taking into account the periodicity along three axes.
    p1 = expand_cell(position_frac, a, b, c, 1, 1, 1)
    p2 = expand_cell(position_frac, a, b, c, 2, 1, 1)
    p3 = expand_cell(position_frac, a, b, c, 1, 2, 1)
    p4 = expand_cell(position_frac, a, b, c, 1, 1, 2)
    p5 = expand_cell(position_frac, a, b, c, 2, 2, 1)
    p6 = expand_cell(position_frac, a, b, c, 2, 1, 2)
    p7 = expand_cell(position_frac, a, b, c, 1, 2, 2)
    p8 = expand_cell(position_frac, a, b, c, 2, 2, 2)

    # constructing the distance matrix and adjacency matrix
    dis_matrix = np.zeros((N, N))
    adj_matrix = np.zeros((N, N))
    for i in range(0, N):
        if i < N - 1:  # if i=N-1,the element of (N,N
            for j in range(i + 1, N):
                array_x = np.vstack((p1[i], p2[i], p3[i], p4[i], p5[i], p6[i], p7[i], p8[i]))
                array_y = np.vstack((p1[j], p2[j], p3[j], p4[j], p5[j], p6[j], p7[j], p8[j]))
                dis_matrix[i, j] = find_min_dis(array_x, array_y)
            inx = find_min_nonzero(dis_matrix[i])
            adj_matrix[i, inx] = 1
    dis_matrix = dis_matrix.T + dis_matrix
    adj_matrix = adj_matrix.T + adj_matrix
    return dis_matrix, adj_matrix

def get_atom_matrix

def prepocess_pos_frac(position_frac, a, b, c):
    dis_matrix, adj_matrix = get_dis_adj_matrix(position_frac, a, b, c)
    # N: number of atoms in unit cell
    N = dis_matrix.shape[0]
    # the elements along the diagonal of the dis_matrix is zero.
    dis_matrix = dis_matrix + np.diag(np.ones(N))
    # element-wise product
    M = np.multiply(adj_matrix, 1 / dis_matrix ** 2)
    pos_discriptor = np.sum(M)
    return pos_discriptor


def get_dis_adj_matrix_error_edition(position_frac, a, b, c):
    N = position_frac.shape[0]  # number of atoms
    # coordinate transformation
    position_frac[:, 0] = position_frac[:, 0] * a
    position_frac[:, 1] = position_frac[:, 1] * b
    position_frac[:, 2] = position_frac[:, 2] * c
    # initializing the distance matrix
    position_1 = np.ones((N, 1)) * position_frac[0]
    distance_square_1 = np.sum((position_1 - position_frac) ** 2, axis=1)
    distance_1 = np.sqrt(distance_square_1)
    dis_matrix = distance_1
    # initializing the adjacency matrix
    adj_matrix = np.zeros((N))
    inx_1 = find_min_nonzero(distance_square_1)
    adj_matrix[inx_1] = 1
    # constructing the distance matrix and adjacency matrix
    for i in range(1, N):
        i_position = np.ones((N, 1)) * position_frac[i]
        distance_square = np.sum((i_position - position_frac) ** 2, axis=1)
        distance = np.sqrt(distance_square)
        dis_matrix = np.vstack((dis_matrix, distance))
        # constructing the  adjacency matrix
        inx = find_min_nonzero(distance)
        adj[inx] = 1
        adj_matrix = np.vstack((adj_matrix, adj))
    return dis_matrix, adj_matrix


file_auid = "/Users/xinming/Documents/aflow/tc_data/auid.npy"
auid = np.load(file_auid)
id = auid[3]
result = search().filter(K.auid == id)
for entry in result:
    ss = entry.positions_fractional
    print(entry.positions_fractional)
file_path = "F:\WORK\AFLOW\data//train/positions_fractional.npy"
pf = np.load(file_path)
ss = pf[3]
dis, adj = get_dis_adj_matrix(ss, 1, 1, 1)
print(dis)
print(adj)
print(dis.shape)
print(adj.shape)
print(adj[3])
