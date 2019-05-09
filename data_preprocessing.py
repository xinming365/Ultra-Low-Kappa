import numpy as np
import matplotlib.pyplot as plt
import os


def split_raw_data():
    # merge the batch_data downloaded from the AFLOW dataset
    save_fp = "F:\WORK\AFLOW\data/train/"
    file_path = "F:\WORK\AFLOW\data/"
    compounds_name = "batch_forces"
    labels_name = "batch_pf"
    total_number = 98
    train_data = np.load(file_path + compounds_name + str(1) + ".npy")  # 初始化
    labels_data = np.load(file_path + labels_name + str(1) + ".npy")  # 初始化
    for index in range(2, total_number + 1):  # 上界不取
        print(index)
        compounds = np.load(file_path + compounds_name + str(index) + ".npy")
        labels = np.load(file_path + labels_name + str(index) + ".npy")
        train_data = np.append(train_data, compounds, axis=0)
        labels_data = np.append(labels_data, labels, axis=0)
    np.save(save_fp + "forces.npy", train_data)
    np.save(save_fp + "positions_fractional.npy", labels_data)


def ls_onehot_encode():
    save_fp = "F:\WORK\AFLOW\data/train/"
    train_data_file = "F:\WORK\AFLOW\data\compounds&labels/train_data.npy"
    train_data = np.load(train_data_file)
    lattice_system_ = []
    l1 = l2 = l3 = l4 = l5 = l6 = l7 = 0
    for i in train_data:
        lattice_system = i[-2]
        if lattice_system == "triclinic\n":  # 三斜晶系
            lattice_system_.append([1, 0, 0, 0, 0, 0, 0])
            l1 = l1 + 1
        if lattice_system == 'monoclinic\n':  # 单斜晶系
            lattice_system_.append([0, 1, 0, 0, 0, 0, 0])
            l2 = l2 + 1
        if lattice_system == 'orthorhombic\n':  # 正交晶系
            lattice_system_.append([0, 0, 1, 0, 0, 0, 0])
            l3 = l3 + 1
        if lattice_system == 'tetragonal\n':  # 四方晶系
            lattice_system_.append([0, 0, 0, 1, 0, 0, 0])
            l4 = l4 + 1
        if lattice_system == 'rhombohedral\n':  # 三角（三方）晶系
            lattice_system_.append([0, 0, 0, 0, 1, 0, 0])
            l5 = l5 + 1
        if lattice_system == 'hexagonal\n':  # 六方晶系
            lattice_system_.append([0, 0, 0, 0, 0, 1, 0])
            l6 = l6 + 1
        if lattice_system == 'cubic\n':  # 立方晶系
            lattice_system_.append([0, 0, 0, 0, 0, 0, 1])
            l7 = l7 + 1
    ls_statistics = [l1, l2, l3, l4, l5, l6, l7]
    np.save(save_fp + "lattice_system.npy", lattice_system_)
    return ls_statistics


def plot_ls_statistics():
    ls_statistics = [2, 41, 935, 985, 2, 1081, 2563]
    N = 7
    index = np.arange(1, N + 1)
    fig, ax = plt.subplots()
    rects = ax.bar(index, height=ls_statistics, color='blue')
    ax.set_xticks(index)
    ax.set_xticklabels(['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'rhombohedral',
                        'hexagonal', 'cuibic'])
    plt.setp(ax.get_xticklabels(), rotation=20)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.ylabel('Count', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.xlabel('Lattice System',fontdict={'family':'Times New Roman','size':12})
    for a, b in zip(index, ls_statistics):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=9, fontproperties="Times New Roman")
    plt.savefig('F:\WORK\AFLOW\data/fig/ls_stastics.png', dpi=300)
    plt.show()


class nspeices_statistics:
    def __init__(self):
        self.nspecies = []

    def n_statistics(self):
        train_data_file = "F:\WORK\AFLOW\data/train/train_data.npy"
        train_data = np.load(train_data_file)
        n1 = n2 = n3 = n4 = n5 = 0
        for i in train_data:
            n = i[-3]
            if n == '1':
                n1 = n1 + 1
            elif n == '2':
                n2 = n2 + 1
            elif n == '3':
                n3 = n3 + 1
            elif n == '4':
                n4 = n4 + 1
            else:
                n5 = n5 + 1
        self.nspecies = [n1, n2, n3, n4, n5]

    def plot_nspecies(self):
        N = 5
        index = np.arange(1, N + 1)
        fig, ax = plt.subplots()
        rects = ax.bar(index, height=self.nspecies, color='blue')
        ax.set_xticks(index)
        ax.set_xticklabels(['singular', 'binary', 'ternary', 'quaternary', '$\geq5$'])
        plt.setp(ax.get_xticklabels(), rotation=20)
        plt.xticks(fontproperties='Times New Roman', size=12)
        plt.yticks(fontproperties='Times New Roman', size=12)
        plt.ylabel('Count', fontdict={'family': 'Times New Roman', 'size': 12})
        # plt.xlabel('Lattice System',fontdict={'family':'Times New Roman','size':12})
        for a, b in zip(index, self.nspecies):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=9, fontproperties="Times New Roman")
        plt.savefig('F:\WORK\AFLOW\data/fig/nspecies_stastics.png', dpi=300)
        plt.show()
