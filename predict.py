from data_preprocessing import get_atom_related_properties, decompose_formula
# from util import get_crystal_property
from data_preprocessing import decompose_formula_II, transform_format
# from util import if_in_the_database
from aflow import *
from Gen_atom import lattice_dict
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt


def get_crystal_property(auid):
    cp_db = json.load(open('./data/resources/crystal_property.json'))
    i = cp_db[auid]
    cp_descriptor = [i['lattice_system_relax'], i['spacegroup_relax'],
                     i['nspecies'], i['natoms'], i['volume_atom'], i['volume_cell'],
                     i['density']]
    return cp_descriptor

# the descriptor comes from the crystal properties and CW properties.

# formula = 'Cu1Sb1S2'

"""
                 
             'Cu1Bi1S2'  'Cu1Sb1S2'   'Cd3As2'    'Bi2Te3'     'Ti1S2'

predicted    1.364          1.43        0.31       1.29         3.19
calculated   0.46           1.44        0.3-0.9    1.3(k_xx)   
experimental 0.5            1.5         0.3         

"""


def ration_equal(x1, x2):
    # x1 and x2 must have the same shape.
    # return the bool value.
    eq = []
    for i in range(len(x1) - 1):
        if x1[0] / x1[i + 1] == x2[0] / x2[i + 1]:
            eq.append(True)
        else:
            eq.append(False)
    return all(eq)


def predict_thermal_conductivity(formula):
    """
    :param formula: string type and standard format. like'Cu1Bi1S2'
    :return: the predicted thermal conductivity based on the XGBoost model.
    """
    CW = get_atom_related_properties(formula=formula)

    # the following codes downloads crystal properties from aflow repository.
    element, element_number = decompose_formula(formula=formula)
    input_dict = {}
    for index, j in enumerate(element):
        input_dict[j] = element_number[index]
    print(input_dict)

    n_species = len(element)
    print(n_species)
    if n_species == 1:
        result = search().filter((K.species == element[0]) & (K.nspecies == n_species)).orderby(K.energy_cell)
    if n_species == 2:
        result = search().filter(
            (K.species == element[0]) & (K.species == element[1]) & (K.nspecies == n_species)).orderby(K.energy_cell)
    if n_species == 3:
        result = search().filter(
            (K.species == element[0]) & (K.species == element[1]) & (K.species == element[2]) & (
                    K.nspecies == n_species)).orderby(
            K.energy_cell)
    if n_species == 4:
        result = search().filter(
            (K.species == element[0]) & (K.species == element[1]) & (K.species == element[2]) & (
                    K.species == element[3])
            & (K.nspecies == n_species)).orderby(K.energy_cell)
    else:
        print("materials of n_species more than 4 are not considered yet.")

    compound_list = []
    property_list = []
    for entry in result:
        property = []
        compound_list.append(entry.raw['compound'])
        property.append(entry.lattice_system_relax)
        property.append(entry.spacegroup_relax)
        property.append(entry.nspecies)
        property.append(entry.natoms)
        property.append(entry.volume_atom)
        property.append(entry.volume_cell)
        property.append(entry.density)
        property_list.append(property)

    target_comp = []
    target_prop = []
    for index, compound in enumerate(compound_list):
        ele, ele_n = decompose_formula(compound)
        input_n = [input_dict[x] for x in ele]
        if ration_equal(input_n, ele_n):
            target_comp.append(compound)
            target_prop.append(property_list[index])

    # print(target_comp)
    # print(target_prop)

    for i in target_prop:
        lattice_system = i[0].strip()
        i[0] = lattice_dict[lattice_system]
    # print(target_prop)
    # [[3, 62, 3, 16, 22.5172, 360.275, 6.20653],
    # [3, 62, 3, 16, 22.5207, 360.332, 6.20556],
    # [3, 62, 3, 16, 22.4966, 359.945, 6.21222],
    # [3, 62, 3, 16, 22.4606, 359.37, 6.22217]]

    # the properties are similar in the list of target_prop. we can choose the one whose energy/cell is the minimum.
    cs = np.array(target_prop[0])
    # cs=[ls,sg,nspecies,natoms_c,v_a,v_c,density]
    # cs=np.array([3, 62, 3, 16, 22.5172, 360.275, 6.20653])

    descriptor = np.concatenate((cs, CW), axis=0)

    descriptor_final = np.reshape(descriptor, newshape=(1, descriptor.shape[0]))
    scaler = pickle.load(file=open(os.path.join('./models', 'scaler_ptc_ab.pkl'), 'rb'))
    descriptor_final = scaler.transform(descriptor_final)
    optimized_Model = pickle.load(file=open('./models/ptc_ab.pkl', 'rb'))
    predict_tc = optimized_Model.predict(descriptor_final)
    return np.exp(predict_tc)


# def train_tc_database(save_file=True):
#     descriptor_final=np.load('./data/train_version_6.npy')
#     tc_final = np.load('./data/train_version_6.npy')
#     result = xgboost_model(train_data=descriptor_final,labels=tc_final,save_models=save_file,model_name='ptc_final.pkl')
#     print(result)
#     return result

def get_descriptor(formula_std, formula_db):
    CW = get_atom_related_properties(formula_std)
    try:
        auid_list = formula_db[formula_std]
    except KeyError:
        print('Error: the {} not found in the ICSD database'.format(formula_std))
        descriptor=[]
    else:
        CP = []
        for auid in auid_list:
            CP_i = get_crystal_property(auid)
            CP.append(CP_i)
        CP = np.mean(CP, axis=0)
        descriptor = np.concatenate((CP, CW), axis=0)
    return descriptor


def load_input_exp_data(exp_database, formula_db):
    """
    :param exp_database: list; a list of tuples, like [('C', 3000)...]
    :param formula_db: dict;
    :return: a list of descriptors generated from the formulas in the exp_database
    """
    descriptor_final = []
    y_data = []
    bool_list = []
    for i in exp_database:
        formula = i[0]
        formula_std = transform_format(formula)
        print(formula_std)
        BOOL = if_in_the_database(formula_std,0)[1]
        descriptor = get_descriptor(formula_std, formula_db)
        if any(descriptor):
            descriptor_final.append(descriptor)
            y_data.append(i[1])
            bool_list.append(BOOL)
    return descriptor_final, y_data, bool_list


def predict_exp_database():
    exp_database = np.load('./data/resources/exp_database.npy',allow_pickle=True)
    formula_db = json.load(open('./data/resources/formula.json'))
    # icsd = json.load(open('./data/resources/icsd.json'))
    x_data, y_data, bool_list = load_input_exp_data(exp_database, formula_db)
    model = pickle.load(file=open(os.path.join('./models', 'ptc_ab.pkl'), 'rb'))
    predict_y = np.exp(model.predict(x_data))
    # the fifth dimension in x_data[0] is the volume/cell
    # xx=np.array(x_data)[:,5]
    In_list = [i for i in range(len(bool_list)) if bool_list[i]]
    Out_list = [i for i in range(len(bool_list)) if not bool_list[i]]
    plt.plot(np.array(y_data)[In_list], predict_y[In_list],'^r-')
    plt.plot(np.array(y_data)[Out_list], predict_y[Out_list], '^k-')
    plt.show()
    return y_data, predict_y,In_list,Out_list


def predict_tc_database(model_name, save_result=True, save_descriptor=False):
    none_tc_db = json.load(open('./data/resources/auid_none_tc.json', 'r'))
    descriptor_final = []
    info_final = []
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
        i = i + 1
        if i % 100 == 0:
            print(i)
    # scaler=pickle.load(file=open(os.path.join('./models','scaler_'+ model_name), 'rb'))
    # scaled_descriptor_final = scaler.transform(descriptor_final)
    # print(type(scaled_descriptor_final))

    # optimized_Model = pickle.load(file=open('./models/ptc_ab.pkl', 'rb'))
    optimized_Model = pickle.load(file=open(os.path.join('./models', model_name), 'rb'))
    predict_tc_log = optimized_Model.predict(descriptor_final)
    predict_tc = np.exp(predict_tc_log)
    print(predict_tc)
    # saved_result= np.concatenate((info_final, predict_tc),axis=1)
    if save_descriptor:
        np.save('./data/train_icsd.npy', descriptor_final)
        # np.save('./data/scaled_train_icsd.npy',scaled_descriptor_final)
    if save_result:
        np.save('./result/pred_tc_log.npy', predict_tc_log)
        np.save('./result/pred_tc.npy', predict_tc)
        # information of the auid and formula corresponding to the kappa
        np.save('./result/pred_tc_with_info.npy', info_final)
    return predict_tc


if __name__ == '__main__':

    # predict_tc = predict_tc_database(model_name='ptc_ab.pkl', save_result=True, save_descriptor=False)
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

    a,b,c,d=predict_exp_database()
    # print(predict_tc)
