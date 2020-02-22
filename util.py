import numpy as np
import os
import json
import copy
from data_preprocessing import decompose_formula
from predict import ration_equal
import pandas as pd


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r_square(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def metric(y_cal, y_pred):
    return r_square(y_cal, y_pred), mae(y_cal, y_pred), rmse(y_cal, y_pred)


def relative_error_percentage(y_true, y_pred, percentage):
    # the percentage of relative error smaller than certain value like 0.3.
    error = np.abs(y_true - y_pred) / np.abs(y_true)
    length = y_true.shape[0]
    count = 0
    for i in error:
        if i < percentage:
            count += 1
    return count / length


def z_score(x):
    return (x - np.mean(x)) / np.std(x)


def min_max(x):
    return x / np.max(x)


def ls_statistics():
    train_data = np.load('./data/raw_data.npy')
    l1 = l2 = l3 = l4 = l5 = l6 = l7 = 0
    ort = []
    tet = []
    hex = []
    cub = []
    for i, j in enumerate(train_data):
        lattice_system = j[-2]
        if lattice_system == "triclinic\n":  # 三斜晶系
            l1 = l1 + 1
        if lattice_system == 'monoclinic\n':  # 单斜晶系
            l2 = l2 + 1
        if lattice_system == 'orthorhombic\n':  # 正交晶系
            l3 = l3 + 1
            ort.append(i)
        if lattice_system == 'tetragonal\n':  # 四方晶系
            l4 = l4 + 1
            tet.append(i)
        if lattice_system == 'rhombohedral\n':  # 三角（三方）晶系
            l5 = l5 + 1
        if lattice_system == 'hexagonal\n':  # 六方晶系
            l6 = l6 + 1
            hex.append(i)
        if lattice_system == 'cubic\n':  # 立方晶系
            l7 = l7 + 1
            cub.append(i)
    ls_statistics = [l1, l2, l3, l4, l5, l6, l7]
    np.save('./data/descriptor/ort.npy', ort)
    np.save('./data/descriptor/tet.npy', tet)
    np.save('./data/descriptor/hex.npy', hex)
    np.save('./data/descriptor/cub.npy', cub)
    return ls_statistics


def save_natoms():
    train_data = pd.read_csv(os.path.join('./data/td.2020.1.29.csv')).to_numpy()
    natoms = train_data[:, 3]
    s = 0
    m = 0
    l = 0
    natoms_S = []
    natoms_M = []
    natoms_L = []
    for index, _ in enumerate(natoms):
        if 0 < _ < 10:
            s = s + 1
            natoms_S.append(index)
        elif 10 <= _ < 20:
            m = m + 1
            natoms_M.append(index)
        else:
            l = l + 1
            natoms_L.append(index)
    np.save('./data/descriptor/natoms_s.npy', natoms_S)
    np.save('./data/descriptor/natoms_m.npy', natoms_M)
    np.save('./data/descriptor/natoms_l.npy', natoms_L)
    return [s, m, l]


def n_statistics():
    # the number of species is no more than 3.
    train_data = np.load('./data/raw_data.npy')
    n1 = n2 = n3 = 0
    nspecies_1 = []
    nspecies_2 = []
    nspecies_3 = []
    for index, i in enumerate(train_data):
        n = i[-3]
        if n == '1':
            n1 = n1 + 1
            nspecies_1.append(index)
        if n == '2':
            n2 = n2 + 1
            nspecies_2.append(index)
        if n == '3':
            n3 = n3 + 1
            nspecies_3.append(index)
    nspecies = [n1, n2, n3]
    np.save('./data/descriptor/1_species.npy', nspecies_1)
    np.save('./data/descriptor/2_species.npy', nspecies_2)
    np.save('./data/descriptor/3_species.npy', nspecies_3)
    return nspecies


def dict(compound):
    pred = np.load('./data/icsd/pred_tc.npy')
    info = np.load('./data/icsd/pred_tc_with_info.npy')
    dict = {}
    for i, v in enumerate(pred):
        dict[info[i][1]] = [info[i][0], pred[i]]
    return dict[compound]


def get_Egap(auid):
    Egap_database = json.load(open('./data/resources/Egap.json'))
    Egap = Egap_database[auid]['Egap']
    if Egap is not None:
        Egap = float(Egap)
    return Egap


def get_auid(formula):
    formula_db = json.load(open('./data/resources/formula.json'))
    return formula_db[formula]


def get_crystal_property(auid):
    cp_db = json.load(open('./data/resources/crystal_property.json'))
    i = cp_db[auid]
    cp_descriptor = [i['lattice_system_relax'], i['spacegroup_relax'],
                     i['nspecies'], i['natoms'], i['volume_atom'], i['volume_cell'],
                     i['density']]
    return cp_descriptor


def screen(n_ele, kappa, Egap):
    pred = np.load('./data/icsd/pred_tc.npy')
    info = np.load('./data/icsd/pred_tc_with_info.npy')
    Egap_database = json.load(open('./data/resources/Egap.json'))
    cmp = info[:, 1]
    cmp_list = []
    for index, j in enumerate(cmp):
        auid = info[index][0]
        E_gap = Egap_database[auid]['Egap']
        if E_gap is not None:
            E_gap = float(E_gap)
        ele_list, num_list = decompose_formula(j)
        total_num = np.sum(num_list)
        if (total_num < n_ele) and (pred[index] < kappa) and (E_gap > Egap):
            cmp_list.append((j, pred[index], auid, E_gap))
    return cmp_list


def if_in_the_database(compound, source_type):
    # source_type: 0:auid_tc.json  1:auid_none_tc.json  2:icsd.json
    resources = {0: './data/resources/auid_tc.json', 1: './data/resources/auid_none_tc.json',
                 2: './data/resources/icsd.json'}
    auid_tc = json.load(open(resources[source_type]))
    _, n_list = decompose_formula(compound)
    retrival = []
    result = []
    for k, v in auid_tc.items():
        _2, n_list2 = decompose_formula(v['compound'])
        if (_ == _2) and (ration_equal(n_list, n_list2)):
            retrival.append((v['compound'], v['agl_thermal_conductivity_300K'], v['auid']))
            result.append(True)
        else:
            result.append(False)
    return retrival, any(result)


def screen_none_tc(n_ele, kappa, Egap):
    """
    screen materials from the none_tc database under certain conditions
    parameters
    ------------
    :param n_ele: int; the number of elements less than n_ele
    :param kappa: float; the thermal conductivity larger than kappa
    :param Egap: float; the bandgap larger than Egap
    :return: list; a list of a series of tuples including the information of compounds
    """
    output = []
    cmp_tuple_list = screen(n_ele, kappa, Egap)
    for _ in cmp_tuple_list:
        retrieval, result = if_in_the_database(_[0], 0)
        if not result:
            print(_)
            output.append(_)
    return output


def load_and_split_descriptor(x, category=0):
    """split the total descriptors into different types

    Args:
        x: ndarray with shape '(N, *)'.
        category: integer.
    Returns:
        train_data with shape '(N, *)'
    """
    train_data = copy.copy(x)
    if category == 0:
        train_data = train_data
    if category == 1:
        train_data = np.delete(train_data, np.arange(0, 7), axis=1)
    if category == 2:
        train_data = np.delete(train_data, np.arange(7, 7 + 25), axis=1)
    if category == 3:
        train_data = np.delete(train_data, np.arange(32, 57), axis=1)
    if category == 4:
        train_data = np.delete(train_data, np.arange(57, 75), axis=1)
    if category == 5:
        train_data = train_data[:, 0:7]
    if category == 6:
        train_data = train_data[:, 7:32]
    if category == 7:
        train_data = train_data[:, 32:57]
    if category == 8:
        train_data = train_data[:, 57:75]
    if category == 9:
        train_data = train_data[:, 0:32]
    return train_data


def read_label(train_data, labels, label_index):
    """Read the specific target label.

    For the kappa, the target label is the logarithm of thermal conductivity.
    For the debye temperature, the target label doesn't change.
    For the Cp and Cv, the target label is the (Cp, Cv)/natoms.
    For the thermal expansion, the target label is the alpha*10^6.

    Args:
        train_data: an array with shape (N, 75), where N is the total number.
        labels: an array with shape (N, 4)
        label_index: a integer.
            0: agl_thermal_conductivity, 1: agl_debye_temperature
            2: Cp at 300K, 3: Cv at 300K, 4: agl_thermal_expansion.
    Returns:
        train_data: ndarray
        y_label: an array with shape (N, 1)
    """
    label = labels[:, label_index]
    natoms = train_data[:, 3]
    if label_index == 0:
        y_label = np.log(label)
    if label_index == 1:
        y_label = label
    if (label_index >= 2) and (label_index <= 3):
        y_label = label / natoms
    if label_index == 4:
        y_label = label * np.power(10, 6)
    return train_data, y_label


if __name__ == '__main__':
    # list = ls_statistics()
    # list = n_statistics()
    # print(list)
    # screen_none_tc(6,0.8,0.1)
    screen_none_tc(7, 0.8, 0.1)
