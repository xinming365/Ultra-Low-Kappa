import numpy as np
import os
import Gen_atom
import re
import copy
from Gen_atom import atomic_dict, lattice_dict
import json

PROPERTY_NUMBER = 25


def find_min_nonzero(array):
    nonzero_array = array[np.nonzero(array)]
    min_value = nonzero_array[np.argmin(nonzero_array)]
    index = list(array).index(min_value)
    return index


def decompose_formula(formula):
    # the standard formula format is like 'XnYnZn', such as Pb1Se1.
    # X/Y/Z is the element. n is the number of element.
    element = re.findall(r'[A-Za-z]+', formula)
    element_number = re.findall(r'(\d+)', formula)
    element_number = [int(i) for i in element_number]
    return element, element_number


def decompose_formula_II(formula):
    # the formula format 'XnYnZn', when n=1, the n is omitted.
    # like PbSe instead of Pb1Se1
    namelist = []
    numlist = []
    ccomps = formula
    while (len(ccomps) != 0):
        stemp = ccomps[1:]
        if (len(stemp) == 0):
            namelist.append(ccomps)
            numlist.append(1.0)
            break
        it = 0
        for st in stemp:
            it = it + 1
            if (st.isupper()):
                im = 0
                for mt in stemp[:it]:
                    im = im + 1
                    if (mt.isdigit()):
                        namelist.append(ccomps[0:im])
                        numlist.append(float(ccomps[im:it]))
                        ccomps = ccomps[it:]
                        break
                    elif (im == len(stemp[:it])):
                        namelist.append(ccomps[0:im])
                        numlist.append(1.0)
                        ccomps = ccomps[it:]
                        break
                break
            elif (it == len(stemp)):
                im = 0
                for mt in stemp:
                    im = im + 1
                    if (mt.isdigit()):
                        namelist.append(ccomps[0:im])
                        numlist.append(float(ccomps[im:]))
                        ccomps = ccomps[it + 1:]
                        break
                    elif (im == len(stemp)):
                        namelist.append(ccomps)
                        numlist.append(1.0)
                        ccomps = ccomps[it + 1:]
                        break
                break
    return namelist, numlist


def transform_format(formula):
    ele_list, num_list = decompose_formula_II(formula)
    formula_2=''
    for index, ele in enumerate(ele_list):
        tmp = ele + str(int(num_list[index]))
        formula_2 = formula_2 + tmp
    return formula_2


def get_z(formula):
    element, element_number = decompose_formula(formula)
    N = np.sum(element_number)
    z = 0
    for i, ele in enumerate(element):
        z = z + atomic_dict[ele] * element_number[i] / N
    return z


def expand_cell(position_frac, a, b, c, n_a, n_b, n_c):
    N = position_frac.shape[0]  # number of atoms
    tmp = copy.copy(position_frac)
    # expand the unit cell. n_a,n_b and n_c represent the number of repetitions along a,b,c axis.
    tmp[:, 0] = (position_frac[:, 0] + (n_a - 1) * np.ones(N)) * a
    tmp[:, 1] = (position_frac[:, 1] + (n_b - 1) * np.ones(N)) * b
    tmp[:, 2] = (position_frac[:, 2] + (n_c - 1) * np.ones(N)) * c
    return tmp


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
        if i < N - 1:  # if i=N-1,the element of (N,N)
            for j in range(i + 1, N):
                array_x = np.vstack((p1[i], p2[i], p3[i], p4[i], p5[i], p6[i], p7[i], p8[i]))
                array_y = np.vstack((p1[j], p2[j], p3[j], p4[j], p5[j], p6[j], p7[j], p8[j]))
                dis_matrix[i, j] = find_min_dis(array_x, array_y)
            inx = find_min_nonzero(dis_matrix[i])
            adj_matrix[i, inx] = 1
    dis_matrix = dis_matrix.T + dis_matrix
    adj_matrix = adj_matrix.T + adj_matrix
    return dis_matrix, adj_matrix


def get_atom_matrix(formula, index):
    # the formula in input parameters is the chemical formula of materials , like "Ag4O2".
    # the index is the index of property list .
    element, element_number = decompose_formula(formula)
    # N : number of atoms in unit cell
    N = np.sum(element_number)
    num_0 = int(element_number[0])
    ele_0 = element[0]
    atom_0 = Gen_atom.atom(ele_0)
    all_property = atom_0.get_property()
    property_0 = all_property[index]
    vec = np.ones((num_0,)) * property_0
    # if there are more elements, we should concatenate these properties.
    if np.shape(element)[0] > 1:
        for i, number in enumerate(element_number, start=0):
            if i > 0:
                num_i = int(number)
                ele_i = element[i]
                atom_i = Gen_atom.atom(ele_i)
                all_property_i = atom_i.get_property()
                property_i = all_property_i[index]
                vec_i = np.ones((num_i,)) * property_i
                vec = np.concatenate((vec, vec_i), axis=0)
    atom_matrix = np.tile(vec, (N, 1))
    atom_matrix_1 = np.abs(atom_matrix.T - atom_matrix)
    atom_matrix_2 = (atom_matrix.T + atom_matrix) / 2
    return atom_matrix_1, atom_matrix_2


def get_descriptor(formula, position_frac, a, b, c, index):
    dis_matrix, adj_matrix = get_dis_adj_matrix(position_frac, a, b, c)
    atom_matrix, atom_matrix_2 = get_atom_matrix(formula, index)
    # N: number of atoms in unit cell
    N = dis_matrix.shape[0]
    # the elements along the diagonal of the dis_matrix is zero.
    # to prevent the reciprocal of 0 becoming infinity, we add one along the diagonal.
    dis_matrix = dis_matrix + np.diag(np.ones(N))
    # element-wise product
    M = np.multiply(adj_matrix, 1 / dis_matrix ** 2)
    T = np.multiply(M, atom_matrix)
    descriptor = np.sum(T)
    return descriptor


def get_atom_related_properties(formula):
    # the data type of the input parameter is string
    element, element_number = decompose_formula(formula)
    sum = np.zeros((PROPERTY_NUMBER,))
    N = np.sum(element_number)
    for i, ele in enumerate(element):
        atom = Gen_atom.atom(ele)
        try:
            ap = atom.get_property()
            tmp = element_number[i] * np.array(ap)
            sum = sum + tmp
        except AttributeError:
            print("No such property!")

    return sum / N


def save_dis_matrix():
    data_path = './data'
    train_data = np.load(os.path.join(data_path, 'train_data.npy'))
    ps = np.load(os.path.join(data_path, 'positions_fractional.npy'))
    save_fp = os.path.join(data_path, 'train_version_2.npz')
    dis_feature = []
    for i, material in enumerate(train_data):
        print(i)
        # formula = material[2]
        # Return geometrical data describing the unit cell in the usual a,b,c,alpha,beta,gamma notation.
        a = float(material[3])
        b = float(material[4])
        c = float(material[5])
        position_frac = list(ps[i])[0]
        dis_matrix, adj_matrix = get_dis_adj_matrix(position_frac, a, b, c)
        # dis_matrix = np.reshape(dis_matrix,(1, dis_matrix.shape[0],dis_matrix.shape[1]))
        dis_feature.append(dis_matrix)
    np.save(save_fp, dis_feature)


def statistics(x):
    max = np.max(x)
    # attention: the min element in the dis_matrix/adj_matrix must be zero. we need to
    # find the non-min value.
    flat_x = x.flatten()
    min = flat_x[find_min_nonzero(flat_x)]
    range = max - min
    mean = np.mean(x)
    std = np.std(flat_x)
    sum = np.sum(x)
    return [mean, min, max, std, range, sum]


def save_structure():
    data_path = './data'
    train_data = np.load(os.path.join(data_path, 'train_data.npy'))
    ps = np.load(os.path.join(data_path, 'positions_fractional.npy'))
    save_fp = os.path.join(data_path, 'train_version_4.npy')
    structure_feature = []
    for i, material in enumerate(train_data):
        print(i)
        # formula = material[2]
        # Return geometrical data describing the unit cell in the usual a,b,c,alpha,beta,gamma notation.
        a = float(material[3])
        b = float(material[4])
        c = float(material[5])
        position_frac = list(ps[i])[0]
        dis_matrix, adj_matrix = get_dis_adj_matrix(position_frac, a, b, c)
        # print(dis_matrix,adj_matrix)
        x = statistics(dis_matrix)
        y = statistics(adj_matrix)
        z = np.concatenate([x, y])
        print(z)
        structure_feature.append(z)
    np.save(save_fp, structure_feature)


def concatenate_all_descriptor():
    data_path = './data'
    train_data = np.load(os.path.join(data_path, 'train_data.npy'))
    ps = np.load(os.path.join(data_path, 'positions_fractional.npy'))
    save_fp = os.path.join(data_path, 'train_version_1.npy')
    final_feature = []
    for i, material in enumerate(train_data):
        print(i)
        formula = material[2]
        # Return geometrical data describing the unit cell in the usual a,b,c,alpha,beta,gamma notation.
        a = float(material[3])
        b = float(material[4])
        c = float(material[5])
        position_frac = list(ps[i])[0]
        # spacegroup, nspecies, natoms, volume_atom, volume_cell, density
        index = [-1, -3, -4, -5, -6, -7]
        crp_list = list(material[index])
        # crystal related properties
        crp = np.array([float(i) for i in crp_list])
        # atom related properties
        arp = get_atom_related_properties(formula)
        # crystal structure fingerprint
        descriptor_vec = []
        for index in range(PROPERTY_NUMBER):
            descriptor = get_descriptor(formula, position_frac, a, b, c, index)
            descriptor_vec.append(descriptor)
        descriptor_vec = np.array(descriptor_vec)
        feature = np.concatenate((crp, arp, descriptor_vec), axis=0)
        final_feature.append(feature)
    np.save(save_fp, final_feature)


def add_new_statistical_feature():
    data_path = './data'
    train_data = np.load(os.path.join(data_path, 'train_version_1.npy'))
    save_fp = os.path.join(data_path, 'train_version_2.npy')
    final_feature = []
    for i, material in enumerate(train_data):
        mean = np.mean(material)
        min = np.min(material)
        max = np.max(material)
        std = np.std(material)
        range = max - min
        statistical_feature = [min, max, range, mean, std]
        new_feature = np.concatenate((material, statistical_feature))
        final_feature.append(new_feature)
    np.save(save_fp, final_feature)


def to_2D():
    data_path = './data'
    raw_train_data = np.load(os.path.join(data_path, 'train_data.npy'))
    train_data = np.load(os.path.join(data_path, 'train_version_1.npy'))
    save_fp = os.path.join(data_path, 'train_version_3.npy')
    num_total_data = train_data.shape[0]
    final_feature = []
    for i, material in enumerate(train_data):
        properties = material[6:]
        mean = np.mean(properties)
        min = np.min(properties)
        max = np.max(properties)
        range = max - min
        std = np.std(properties)
        sum = np.sum(properties)
        formula = raw_train_data[i][2]
        z = get_z(formula)
        lattice_system = raw_train_data[i][-2].strip()
        ls = lattice_dict[lattice_system]
        mat_list = list(material)
        mat_list.insert(0, ls)
        mat_list.insert(0, z)
        statistical_feature = [min, max, range, sum, mean, std]
        new_feature = np.concatenate((mat_list, statistical_feature))
        new_feature = new_feature.reshape(8, 8)
        final_feature.append(new_feature)
    final_feature = np.reshape(final_feature, (num_total_data, 8, 8, 1))
    print(final_feature.shape)
    np.save(save_fp, final_feature)


def concatenate_2D_with_structure_feature():
    data_path = './data'
    raw_train_data = np.load(os.path.join(data_path, 'train_data.npy'))
    train_data = np.load(os.path.join(data_path, 'train_version_1.npy'))
    structure_data = np.load(os.path.join(data_path, 'train_version_4.npy'))
    save_fp = os.path.join(data_path, 'train_version_5.npy')
    num_total_data = train_data.shape[0]
    final_feature = []
    for i, material in enumerate(train_data):
        properties = material[6:]
        mean = np.mean(properties)
        min = np.min(properties)
        max = np.max(properties)
        range = max - min
        std = np.std(properties)
        sum = np.sum(properties)
        formula = raw_train_data[i][2]
        lattice_system = raw_train_data[i][-2].strip()
        ls = lattice_dict[lattice_system]
        mat_list = list(material)
        mat_list.insert(0, ls)
        statistical_feature = [min, max, range, sum, mean, std]
        structure_feature = structure_data[i]
        new_feature = np.concatenate((mat_list, statistical_feature, structure_feature))
        final_feature.append(new_feature)
    print(len(final_feature))
    np.save(save_fp, final_feature)


def save_tc_database():
    tc_db = json.load(open('./data/auid_tc.json', 'r'))
    descriptor_final = []
    tc_final = []
    for k, v in tc_db.items():
        CW = get_atom_related_properties(v['compound'])
        ls = lattice_dict[v['lattice_system_relax']]
        cs = [ls, float(v['spacegroup_relax']), float(v['nspecies']), float(v['natoms']),
              float(v['volume_atom']), float(v['volume_cell']), float(v['density'])]
        descriptor = np.concatenate((cs, CW), axis=0)
        descriptor_final.append(descriptor)
        tc_final.append(float(v['agl_thermal_conductivity_300K']))
    descriptor_final = np.array(descriptor_final)
    np.save('./data/train_version_6.npy', descriptor_final)
    np.save('./data/labels_6.npy', np.log(tc_final))


def save_tc_database(save_file=True):
    none_tc_db = json.load(open('./data/auid_tc.json', 'r'))
    descriptor_final = []
    info_final = []
    tc_final = []
    i = 0
    for k, v in none_tc_db.items():
        formula = v['compound']
        CW = get_atom_related_properties(formula)
        ls = lattice_dict[v['lattice_system_relax']]
        cs = [ls, float(v['spacegroup_relax']), float(v['nspecies']), float(v['natoms']),
              float(v['volume_atom']), float(v['volume_cell']), float(v['density'])]
        descriptor = np.concatenate((cs, CW), axis=0)
        descriptor_final.append(descriptor)
        info_final.append([v['auid'], formula])
        tc_final.append(float(v['agl_thermal_conductivity_300K']))
        i = i + 1
        if i % 100 == 0:
            print(i)
    if save_file:
        np.save('./data/train_version_7.npy', descriptor_final)
        np.save('./data/train_tc_with_info.npy', info_final)
        np.save('./data/label_7.npy', tc_final)


def save_Egap():
    database = json.load(open('./data/resources/icsd.json'))
    dict = {}
    for k, v in database.items():
        Egap_dict = {}
        Egap_dict['Egap'] = v['Egap']
        Egap_dict['Egap_fit'] = v['Egap_fit']
        Egap_dict['Egap_type'] = v['Egap_type']
        dict[v['auid']] = Egap_dict
    json.dump(dict, open('./data/resources/Egap.json', 'w'))


def save_crystal_property():
    database = json.load(open('./data/resources/icsd.json'))
    dict1 = {}
    for k, v in database.items():
        dict2 = {'lattice_system_relax': lattice_dict[v['lattice_system_relax']],
                 'spacegroup_relax': float(v['spacegroup_relax']), 'nspecies': float(v['nspecies']),
                 'natoms': float(v['natoms']), 'volume_atom': float(v['volume_atom']),
                 'volume_cell': float(v['volume_cell']), 'density': float(v['density'])}
        dict1[v['auid']] = dict2
    json.dump(dict1,open('./data/resources/crystal_property.json','w'))



def formula_database():
    formula_db = {}
    database = json.load(open('./data/resources/icsd.json'))
    for k, v in database.items():
        formula_db.setdefault(v['compound'], []).append(v['auid'])
    json.dump(formula_db, open('./data/resources/formula.json', 'w'))


if __name__ == '__main__':
    # concatenate_all_descriptor()
    # add_new_statistical_feature()
    # to_crystal_image()
    # train_data = np.load('./data/train_data.npy')
    # formula = train_data
    # get_z(formula)
    # to_2D()
    # concatenate_2D_with_structure_feature()
    save_tc_database()
    # data_path = './data'
    # train_data = np.load(os.path.join(data_path, 'train_data.npy'))
    # num_total_data = train_data.shape[0]
    # ps = np.load(os.path.join(data_path, 'positions_fractional.npy'))
    # save_fp = os.path.join(data_path, 'train_version_4.npy')
    # final_feature = []
    # i = 5485
    # material = train_data[i]
    # formula = material[2]
    # # Return geometrical data describing the unit cell in the usual a,b,c,alpha,beta,gamma notation.
    # a = float(material[3])
    # b = float(material[4])
    # c = float(material[5])
    # position_frac = list(ps[i])[0]
    # dis_matrix, adj_matrix = get_dis_adj_matrix(position_frac, a, b, c)
    # atom_matrix, atom_matrix_2 = get_atom_matrix(formula, 10)
    # # print(atom_matrix_2)
    # N: number of atoms in unit cell
    # image_matrix = np.reshape(atom_matrix_2 , (N, N, 1))
    # spp = SpatialPyramidPooling([1, 2])
    # outputs = spp.call(image_matrix)
    # print(outputs)
    save_tc_database()
