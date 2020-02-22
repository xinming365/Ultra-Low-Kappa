import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Colormap
from matplotlib import colors
import pickle
from util import min_max
from data_preprocessing import decompose_formula
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.font_manager import FontProperties
from analysis import significance_analysis
import pandas as pd

left_limit = 0
right_limit = 300
x_label = 'Calculated Thermal Conductivity'
y_label = 'Predicted Thermal Conductivity'

color = 'blue'
if not os.path.exists('./fig'):
    os.makedirs('./fig')
fig_path = './fig'


def krr_plot(y, y_, fig_name):
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y, y_, s=2, c=color, alpha=0.8, label='Train')
    equal_x = np.arange(left_limit, right_limit, 0.1)
    plt.xlim(left_limit, right_limit)
    plt.ylim(left_limit, right_limit)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.plot(equal_x, equal_x, color='k', linestyle='dashed', linewidth=1, markersize=1)
    plt.savefig(os.path.join(fig_path, fig_name))


def dnn_plot(y, y_, fig_name):
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y, y_, s=2, c=color, alpha=0.8, label='Train')
    equal_x = np.arange(left_limit, right_limit, 0.1)
    plt.xlim(left_limit, right_limit)
    plt.ylim(left_limit, right_limit)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.plot(equal_x, equal_x, color='k', linestyle='dashed', linewidth=1, markersize=1)
    plt.savefig(os.path.join(fig_path, fig_name))


def scatter_hist(x, y, **fig_name):
    # definitions for the axes
    left, width = 0.1, 0.72
    bottom, height = 0.1, 0.72
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.13]
    rect_histy = [left + width + spacing, bottom, 0.13, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8), dpi=400)
    xmajorLocator = MultipleLocator(1)

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.set_xlabel('Calculated ln[$\kappa$]', fontdict={'family': 'Times New Roman', 'size': 20})
    ax_scatter.set_ylabel('Predicted ln[$\kappa$]', fontdict={'family': 'Times New Roman', 'size': 20})
    ax_scatter.tick_params(direction='in', top=True, right=True, labelsize=18, size=5)
    ax_scatter.xaxis.set_major_locator(xmajorLocator)
    ax_scatter.yaxis.set_major_locator(xmajorLocator)
    labels = ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    ax_scatter.scatter(x, y, edgecolors='k', linewidths=0.1, color='#90a0c7')
    # sandybrown
    # now determine nice limits by hand:
    binwidth = 0.25
    l_lim = np.floor(np.array([x, y]).min() / binwidth) * binwidth - 2 * binwidth
    r_lim = np.ceil(np.array([x, y]).max() / binwidth) * binwidth + 2 * binwidth
    ax_scatter.set_xlim((l_lim, r_lim))
    ax_scatter.set_ylim((l_lim, r_lim))
    ax_scatter.grid(linestyle='-.', alpha=.45)
    ax_scatter.set_axisbelow(True)
    # plot the ypred=ytest dashed line.
    equal_x = np.arange(l_lim, r_lim, 0.1)
    ax_scatter.plot(equal_x, equal_x, color='k', linestyle='dashed', linewidth=1, markersize=1)
    # ax_scatter.grid(linestyle='--')

    # bins = np.arange(l_lim, r_lim + binwidth, binwidth)
    bins = 30
    sns.distplot(x, bins=bins, hist=True, kde=True, ax=ax_histx)
    # ax_histx.hist(x, bins=bins, histtype='stepfilled',color='#b2b2b6',alpha=0.7)
    ax_histx.axis('off')
    sns.distplot(y, bins=bins, hist=True, kde=True, ax=ax_histy, vertical=True)
    # ax_histy.hist(y, bins=bins, orientation='horizontal',alpha=0.7,color='#b2b2b6')
    ax_histy.axis('off')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    plt.text(x=0.2, y=0.8, s=r'$R^2$=0.902', fontdict={'family': 'Times New Roman', 'size': 20}
             , transform=ax_scatter.transAxes)

    save_fig = True
    if save_fig:
        plt.savefig(os.path.join(fig_path, fig_name['fig_name']), format='eps', bbox_inches='tight')
    plt.show()


def plot_statistics(save=False):
    # count the number of materials belong to different lattice systems.
    ls_statistics = [2, 39, 923, 975, 2, 1065, 2480]
    N = 7
    index = np.arange(1, N + 1)
    COLOR = '#7daccc'
    plt.figure(figsize=(8.6, 4), dpi=300)
    ax1 = plt.subplot(1, 2, 1)
    ax1.grid(linestyle='-.', alpha=.45)
    ax1.set_axisbelow(True)
    ax1.bar(index, height=ls_statistics, color=COLOR)
    ax1.set_xticks(index)
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, fontproperties='Times New Roman', fontsize=15)
    # ax1.set_xticklabels(['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'rhombohedral',
    #                     'hexagonal', 'culbic'])
    ax1.set_xticklabels(['tri', 'mon', 'ort', 'tet', 'rho',
                         'hex', 'cub'])
    # plt.setp(ax1.get_xticklabels(), rotation=20)
    # ax1.tick_params(axis='both', labelsize=13)
    ax1.set_ylabel('Number', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.xlabel('Lattice System', fontdict={'family': 'Times New Roman', 'size': 13})
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    for a, b in zip(index, ls_statistics):
        ax1.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=9, fontproperties="Times New Roman")

    data = np.load('./data/labels.npy')
    tc = data[:, 0]
    tc = np.log(tc)
    ax2 = plt.subplot(1, 2, 2)
    ax2.grid(linestyle='-.', alpha=.45)
    ax2.set_axisbelow(True)
    ax2.hist(x=tc, bins=40, color=COLOR)
    ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, fontproperties='Times New Roman', fontsize=15)
    ax2.set_xlabel('$\log(\kappa,W m^{-1} K^{-1})$', fontproperties='Times New Roman', size=13)
    # ax2.set_ylabel('Number',fontproperties='Times New Roman',size=13)
    # plt.savefig('./fig/tc_hist.png', dpi=300, bbox_inches='tight')
    # plt.show()
    save = True
    if save:
        plt.savefig('./fig/stastics2.eps', dpi=300, bbox_inches='tight')
    plt.show()


def plot_nspecies(save=False):
    # count the number of materials of singular/binary/ternary
    n_species = [98, 2481, 2907, 0]
    N = len(n_species)
    index = np.arange(1, N + 1)
    fig, ax = plt.subplots()
    ax.bar(index, height=n_species, color='blue')
    ax.set_xticks(index)
    ax.set_xticklabels(['singular', 'binary', 'ternary', '$\geq4$'])
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.ylabel('Count', fontdict={'family': 'Times New Roman', 'size': 13})
    # plt.xlabel('Lattice System',fontdict={'family':'Times New Roman','size':12})
    for a, b in zip(index, n_species):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=9, fontproperties="Times New Roman")
    if save:
        plt.savefig('./fig/nspecies_stastics.png', dpi=600, bbox_inches='tight')
    plt.show()



def tc_barchart():
    labels = ['-Crystal Part', '-CW', '-Structure', '-Statistical part', 'Crystal Part', 'CW', 'Structure',
              'Statistical Part', 'Crystal + CW', 'All Descriptor']
    # xgboost_result=[2.51, 3.03, 2.17, 2.23, 3.84, 3.41, 4.09, 4.11, 2.13, 2.29]
    # xgboost_result = [2.51, 3.03, 2.17, 2.23, 3.84, 3.41, 4.09, 4.11, 2.15, 2.29]
    # krr_result = [2.83, 3.33, 2.54, 2.28, 4.50, 3.53, 4.40, 4.32, 2.35, 2.47]
    # svr_result = [2.68, 3.56, 2.55, 2.66, 4.33, 3.63, 4.65, 4.47, 2.66, 2.57]
    # dnn_result = [2.34, 2.91, 2.46, 2.34, 4.06, 3.51, 4.07, 4.28, 2.27, 2.26]
    xgboost_result = [2.5, 3.0, 2.2, 2.2, 3.8, 3.4, 4.1, 4.1, 2.2, 2.3]
    krr_result = [2.8, 3.3, 2.5, 2.3, 4.5, 3.5, 4.4, 4.3, 2.4, 2.5]
    svr_result = [2.7, 3.6, 2.6, 2.7, 4.3, 3.6, 4.7, 4.5, 2.7, 2.6]
    dnn_result = [2.3, 2.9, 2.5, 2.3, 4.1, 3.5, 4.1, 4.3, 2.3, 2.3]

    x = np.arange(len(labels))
    width = 0.15
    ymajorLocator = MultipleLocator(base=1)

    fig, ax = plt.subplots(dpi=300)
    rects1 = ax.bar(x - 2 * width, height=svr_result, width=width, label='SVR', edgecolor='k', linewidth=0.3,
                    alpha=0.85)
    rects2 = ax.bar(x - width, height=krr_result, width=width, label='KRR', edgecolor='k', linewidth=0.3, alpha=0.85)
    rects3 = ax.bar(x, height=xgboost_result, width=width, label='XGBoost', edgecolor='k', linewidth=0.3, alpha=0.85)
    rects4 = ax.bar(x + width, height=dnn_result, width=width, label='FC', edgecolor='k', linewidth=0.3, alpha=0.85,
                    color='brown')

    ax.set_ylabel('MAE of $\kappa$ ($Wm^{-1}K^{-1}$)', fontproperties='Times New Roman', size=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties='Times New Roman'
                       , rotation=20)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.set_ylim(bottom=0, top=5.5)
    ax.axhline(y=2.29, ls='-.', color='k', alpha=0.8, linewidth=0.6)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontproperties='Times New Roman', size=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    fig.set_size_inches(18, 4)
    plt.tick_params(axis='x', length=0)
    plt.tick_params(axis='y', direction='in')
    # plt.xticks(x,labels,rotation=20)
    ax.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=16)
    # plt.show()
    SAVE = False
    if SAVE:
        plt.savefig('./fig/contrast2.png', bbox_inches='tight')
    plt.show()


def tc_barchart_II():
    labels = ['All categories', '-Crystal part', '-CW', '-Structure', '-Statistical part', 'Crystal part', 'CW', 'Structure',
              'Statistical part', 'Crystal + CW']
    pd.read_csv(open('./result/descriptors_analysis.csv','r'))

    xgboost_result = [2.24, 2.52, 2.93, 2.91, 2.23, 3.84, 3.41, 4.09, 4.15, 2.13]
    krr_result = [2.47, 2.83, 3.33, 2.54, 2.28, 4.50, 3.53, 4.40, 4.33, 2.35]
    svr_result = [2.57, 2.68, 3.56, 2.55, 2.66, 4.33, 3.63, 4.65, 4.47, 2.66]
    dnn_result = [2.26, 2.34, 2.91, 2.46, 2.34, 4.06, 3.51, 4.07, 4.28, 2.27]

    x = np.arange(len(labels))
    width = 0.15
    ymajorLocator = MultipleLocator(base=1)
    text_size=23

    fig, ax = plt.subplots(dpi=300)
    rects1 = ax.bar(x - 2 * width, height=svr_result, width=width, label='SVR', edgecolor='k', linewidth=0.3,
                    alpha=0.85)
    rects2 = ax.bar(x - width, height=krr_result, width=width, label='KRR', edgecolor='k', linewidth=0.3, alpha=0.85)
    rects3 = ax.bar(x, height=xgboost_result, width=width, label='XGBoost', edgecolor='k', linewidth=0.3, alpha=0.85)
    rects4 = ax.bar(x + width, height=dnn_result, width=width, label='FC', edgecolor='k', linewidth=0.3, alpha=0.85,
                    color='brown')

    ax.set_ylabel('MAE of $\kappa$ (W/mK)', fontproperties='Times New Roman', size=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties='Times New Roman'
                       , rotation=20)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.set_ylim(bottom=0, top=5.5)
    ax.axhline(y=2.24, ls='-.', color='k', alpha=0.8, linewidth=0.6)
    ax.legend(fontsize=16)

    fig.set_size_inches(18, 4)
    plt.tick_params(axis='x', length=0)
    plt.tick_params(axis='y', direction='in')
    # plt.xticks(x,labels,rotation=20)
    ax.tick_params(axis='y', labelsize=text_size)
    plt.tick_params(axis='x', labelsize=text_size)
    plt.savefig('./fig/fig1.eps', bbox_inches='tight')
    plt.show()


def get_importance_llist(type):
    # type=1: feature importance based on the splits used in the xgboost.
    # type=2: changing one but others invariant
    if type==1:
        optimized_GBM = pickle.load(file=open(os.path.join('./models', 'ptc_ab.pkl'), 'rb'))
        xgb = optimized_GBM.best_estimator_
        feature_importances = xgb.feature_importances_
    if type==2:
        feature_importances = significance_analysis()

    return feature_importances


def plot_feature_importance(feature_importances):
    feature = ['lattice type', 'space group', 'nspecies', 'natoms', 'volume/atom', 'volume/cell', r'$\rho$',
               'atomic number', 'mendeleev number', 'period', 'group', 'mass',
               'density', 'valence', 'radii', '$r_{cov}$', '$r_{vdw}$', 'electron affinity', 'electron negativity',
               'ionization energy', '$T_b$', '$T_m$', 'molar volume', 'thermal conductivity', 'orbital exponent',
               'polarizability', 'global hardness', 'electrophilicity', r'$\Delta H_{atomic}$', 'fusion enthalpy',
               'vaporization enthalpy', 'binding energy']
    pos = range(len(feature_importances))
    sorted_index = np.argsort(feature_importances)
    fig, ax1 = plt.subplots(figsize=(7, 8))
    ax1.xaxis.grid(True, linestyle='-.', which='major',
                   color='grey')
    ax1.tick_params(labelsize=12)
    # plt.grid(True, axis='x',which='major', ls='-.')
    sorted_feature = feature_importances[sorted_index]
    min_max_color = min_max(sorted_feature)
    min_max_color = np.power(min_max_color, 1 / 2)
    print(min_max_color)
    bar_color = [tuple([0.1803921568627451, 0.3411764705882353, 0.5019607843137255, i]) for i in min_max_color]
    # bar_color = [tuple([0.49019607843137253, 0.6745098039215687, 0.8, i]) for i in min_max_color]
    ax1.barh(pos, sorted_feature, align='center', color=bar_color)
    FP = FontProperties(family='Times New Roman', size=14)
    plt.yticks(pos, np.array(feature)[sorted_index], fontproperties=FP)
    plt.xticks(fontproperties=FP)

    plt.xlabel('Importance', size=18, fontdict={'family': 'Times New Roman'})
    SAVE = False
    if SAVE:
        plt.savefig('./fig/feature_importance.png', dpi=600, bbox_inches='tight')
    plt.show()


def info_with_the_element(element):
    train_icsd = np.load('./data/train_icsd.npy')
    # tc=np.load('./result/pred_tc_log.npy')
    tc = np.load('./result/pred_tc.npy')
    tc_info = np.load('./result/pred_tc_with_info.npy')
    descriptor_with_the_element = []
    tc_with_the_element = []
    for index, info in enumerate(tc_info):
        # print(index)
        formula = info[1]
        ele, ele_number = decompose_formula(formula)
        if element in ele:
            descriptor_with_the_element.append(list(train_icsd[index]))
            tc_with_the_element.append(tc[index])
    descriptor_with_the_element = np.array(descriptor_with_the_element)

    return descriptor_with_the_element, tc_with_the_element


def plot_element_dependence(element):
    descriptor_with_the_element, tc_with_the_element = info_with_the_element(element)
    startcolor = 'r'
    midcolor = 'y'
    endcolor = 'g'
    cmap1 = colors.LinearSegmentedColormap.from_list('own2', [startcolor, midcolor, endcolor])
    norm = mpl.colors.Normalize(vmin=-3, vmax=6)
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(descriptor_with_the_element) != 0:
        # two axis: '$\Delta H_{atomic}$' and density
        plt.scatter(descriptor_with_the_element[:, -4], descriptor_with_the_element[:, 6], c=tc_with_the_element
                    , s=2, cmap=cmap1, norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.text(0.1, 0.8, element, transform=ax.transAxes, fontsize=72)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # ax.set_aspect('auto')
    # cb=plt.colorbar()
    # cb.ax.set_ylabel('Count',fontsize=13)
    SAVE = True
    if SAVE:
        path = './fig/element_dependence'
        if not os.path.exists(path):
            os.makedirs(path)
        figname = element + '_dependence.png'
        plt.savefig(os.path.join(path, figname), dpi=600, pad_inches=0)
    plt.show()


def plot_color_bar():
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_axes([0.025, 0.7, 0.95, 0.15])
    startcolor = 'r'
    midcolor = 'y'
    endcolor = 'g'
    cmap = colors.LinearSegmentedColormap.from_list('own2', [startcolor, midcolor, endcolor])
    norm = mpl.colors.Normalize(vmin=-3, vmax=6)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal'
                                    )
    size = 50
    cb1.set_label(r'The predicted ln[$\kappa$ (W/(mK))]', fontdict={'family': 'Times New Roman', 'size': size})
    ax1.tick_params(direction='in', labelsize=40, length=10, width=2, pad=20)
    plt.savefig(os.path.join('./fig', 'colorbar.eps'), dpi=600, format='eps', pad_inches=0)
    plt.show()


def plot_accuracy(result):
    pass


def plot_acc_on_partial_feature():
    svr_acc = np.load('./result/accuracy/accuracy_partial_random_svr.npy')
    krr_acc = np.load('./result/accuracy/accuracy_partial_random_krr.npy')
    xgboost_acc = np.load('./result/accuracy/accuracy_partial_random_xgboost.npy')
    y_1=svr_acc[:,0]
    y_2=krr_acc[:,0]
    y_4=xgboost_acc[:,0]
    x = np.arange(len(y_1))
    yerr_p= svr_acc[:,1]
    yerr_m=svr_acc[:,2]
    # yerr_array=[yerr_m-y,yerr_p-y]
    # plt.errorbar(np.arange(len(y)),y,yerr_array,marker='s')
    plt.plot(x, y_1, 'o-', label='SVR')
    plt.plot(x, y_2, 'x-', label='KRR')
    # plt.plot(x, y_3, 'D-', label='FC')
    plt.plot(x, y_4, 's-', label='XGBoost')
    plt.legend()
    plt.show()
    plt.xlabel('dimensionality')
    plt.ylabel('RMSE of log($\kappa$, (W/mK))')
    # result = np.load('./result/accuracy/accuracy_partial.npy')
    # y = result[:, 0]
    # y_2 = np.load('./result/accuracy/accuracy_partial_unsort.npy')[:, 0]
    # x = range(len(y))
    # plt.plot(x, y, 'bo-',label='sort' )
    # plt.plot(x, y_2,'r*-',label='unsort')
    # plt.show()


def plot_acc_on_category():
    # accuracy on the category of lattice system.
    svr_acc = [0.8866525324888326, 0.7374347851829923, 0.7769923700740443, 0.6480240946879124]
    krr_acc = [0.9093637336518622, 0.8078166601452073, 0.8923666273338905, 0.7439852928100251]
    fc_acc = [0.8918945552705899, 0.8135292925720039, 0.8719702783812269, 0.7202942825594977]
    xgboost_acc = [0.9173571560911747, 0.8275425824837026, 0.8832342927156365, 0.7233761844743497]
    y_1 = np.array(svr_acc)
    y_2 = np.array(krr_acc)
    y_3 = np.array(fc_acc)
    y_4 = np.array(xgboost_acc)
    x = range(len(y_1))
    # accuracy on the category of atoms/cell.(0-10; 10-20; 20-)
    r2_svr = np.array([0.7729204878212963, 0.7890204851492184, 0.8772619912007676])
    r2_krr = np.array([0.8212165862855878, 0.865736823141431, 0.8929842190033598])
    r2_fc = np.array([0.8037130609557539, 0.8629226426419969, 0.8718019511779229])
    r2_xgboost = np.array([0.81913451,0.84766157,0.86219186])
    x2 = range(len(r2_svr))
    ax1 = plt.subplot(121)
    ax1.set_ylim(0.5,1)
    ax1.set_xticklabels(['ort','tet','hex','cub'])
    xmajorLocator = MultipleLocator(1)
    ax1.xaxis.set_major_locator(xmajorLocator)
    # ax1.grid(True, linestyle='-')
    plt.plot(x, y_1, 'o-', label='SVR')
    plt.plot(x, y_2, 'x-', label='KRR')
    plt.plot(x, y_3, 'D-', label='FC')
    plt.plot(x, y_4, 's-', label='XGBoost')
    plt.legend()

    ax2=plt.subplot(122, sharey=ax1)
    ax2.set_xticklabels(['small', 'middle', 'large'])
    plt.plot(x2, r2_svr, 'o-', label='SVR')
    plt.plot(x2, r2_krr, 'x-', label='KRR')
    plt.plot(x2, r2_fc, 'D-', label='FC')
    plt.plot(x2, r2_xgboost, 's-', label='XGBoost')
    plt.legend()
    plt.show()


def plot_table(type):
    # type=0: 5*2; type=1:5*7
    startcolor = 'r'
    midcolor = 'y'
    endcolor = 'g'
    cmap1 = colors.LinearSegmentedColormap.from_list('own2', [startcolor, midcolor, endcolor])
    norm = mpl.colors.Normalize(vmin=-4.5, vmax=6)
    if type == 0:
        cases = [['Li', 'Be'], ['Na', 'Mg'], ['K', 'Ca'],
                 ['Rb', 'Sr'], ['Cs', 'Ba']]
        fig_size = (2, 5)
        fig_name = 'table_1.eps'
    if type == 1:
        cases = [['', '', 'B', 'C', 'N', 'O', 'F'],
                 ['', '', 'Al', 'Si', 'P', 'S', 'Cl'],
                 ['Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br'],
                 ['Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I'],
                 ['Au', 'Hg', 'Tl', 'Pb', 'Bi', '', '']]
        fig_size = (7, 5)
        fig_name = 'table_2.eps'

    if type == 2:
        cases = [['H']]
        fig_size = (1, 1)
        fig_name = 'table_example.eps'

    if type == 3:
        cases = [['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni'],
                 ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd'],
                 ['La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt'],
                 ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', ' ', ' ']]
        fig_size = (8, 4)
        fig_name = 'tabl_3.eps'

    if type == 4:
        cases = [['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                  'Yb', 'Lu'],
                 ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
                  'Fm', 'Md', 'No', 'Lr']]
        fig_size = (14, 2)
        fig_name = 'table_4.eps'
    rows = len(cases)
    cols = len(cases[0])
    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    gs1 = gridspec.GridSpec(rows, cols, figure=fig)
    gs1.update(wspace=0, hspace=0)  # set the spacing between axes.

    for col in range(cols):
        print(col)
        for row in range(rows):
            print(row)
            case = cases[row][col]
            print(case)
            descriptor_with_the_element, tc_with_the_element = info_with_the_element(case)
            ax1 = fig.add_subplot(gs1[row, col])
            plt.axis('on')
            if case == '':
                ax1.axis('off')
            else:
                if tc_with_the_element != []:
                    ax1.text(0.1, 0.8, case, transform=ax1.transAxes, fontsize=12)
                    ax1.scatter(descriptor_with_the_element[:, -4], descriptor_with_the_element[:, 6],
                                c=tc_with_the_element
                                , s=0.1, cmap=cmap1, norm=norm)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_xticks([])
            ax1.set_yticks([])
            fig.add_subplot(ax1)

    if type == 2:
        plt.xlabel(r'$\Delta H_{atomic}$')
        plt.ylabel(r'$\rho$')
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join('./fig', fig_name), dpi=300, format='eps', bbox_inches='tight')
    plt.show()


def hh_comp():
    tc_cal = [16.7, 62.1, 1.24, 20, 37.1, 27.7, 22.5, 27.1, 17.8, 36.2, 27.2, 29.1, 21.9, 30.1, 20.7,
              23.3, 24.4, 36.9, 22.7, 19.8, 47.6, 32.9, 32.8, 29.1, 31.2, 24.1, 26.2, 16.5, 30.1, 17.4,
              12.8, 33.0, 37.2, 30.0, 24.7, 19.8, 22.1, 17.5, 14.3, 10.6, 22.9, 19.6, 25.3, 21.1, 19.5,
              15.2, 16.8, 17.5, 24.8, 28.8, 6.05, 9.95, 18.2, 15.1, 10.3, 32.3, 26.7, 15.9, 1.72, 33.1,
              27.1, 12.8, 13.0, 13.0, 2.84, 15.7, 20.3, 43.7, 33.4, 22.7, 20.9, 21.3, 3.49, 20.9, 6.44]

    tc_pred = [4.6, 34.0, 0.1, 5.1, 13.1, 10.9, 5.9, 5.5, 7.5, 7.1, 6.6, 4.5, 6.0, 15.3, 8.9, 10.7, 7.5,
               7.7, 9.7, 8.2, 9.7, 4.8, 6.3, 11.4, 8.6, 8.6, 7.4, 9.3, 6.2, 6.5, 5.6, 6.0, 5.7, 6.0, 5.9,
               6.2, 5.7, 8.3, 7.1, 9.8, 11.5, 1.5, 4.7, 3.9, 6.3, 3.7, 5.4, 7.6, 4.4, 4.3, 8.0, 5.4]
    text = ['AuAIHf', 'BLiSi', 'BiBaK', 'CoAsHf', 'CoAsTi', 'CoAsZr', 'CoBiHf', 'CoBiTi', 'CoBiZr',
            'CoGeNb', 'CoGeTa', 'CoGeV', 'CoHfSb', 'CoNbSi', 'CoNbSn', 'CoSbTi', 'CoSbZr', 'CoSiTa',
            'CoSnTa', 'CoSnV', 'FeAsNb', 'FeAsTa', 'FeGeW', 'FeNbSb', 'FeSbTa', 'FeSbV', 'FeTeTi',
            'GeAILi', 'IrAsTi', 'IrAsZr', 'IrBiZr', 'IrGeNb', 'IrGeTa', 'IrGeV', 'IrHfSb', 'IrNbSn',
            'IrSnTa', 'NiAsSc', 'NiBiSc', 'NiBiY', 'NiGaNb', 'NiGeHf', 'NiGeTi', 'NiGeZr', 'NiHfSn',
            'NiPbZr', 'NiSnTi', 'NiSnZr', 'OsNbSb', 'OsSbTa', 'PCdNa', 'PdBiSc', 'PdGeZr', 'PdHfSn',
            'PdPbZr', 'PtGaTa', 'PtGeTi', 'PtGeZr', 'PtLaSb', 'RhAsTi', 'RhAsZr', 'RhBiHf', 'RhBiTi',
            'RhBiZr', 'RhLaTe', 'RhNbSn', 'RhSnTa', 'RuAsNb', 'RuAsTa', 'RuNbSb', 'RuSbTa', 'RuTeZr',
            'SbNaSr', 'SiAILi', 'ZnLiSb']


if __name__ == '__main__':
    # save_fig=True
    # ypred = np.load('./result/ypred_xgboost2.npy')
    # ytest = np.load('./result/ytest2.npy')
    # scatter_hist(ytest, ypred, fig_name='fig_2.eps')
    # plot_statistics(save=True)
    # plot_nspecies(save=False)
    # tc_hist()
    # tc_barchart()
    tc_barchart_II()
    # pca()
    # plot_feature_importance()

    # plot_table(3)
    #
    # table1 = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
    #        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
    #       'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
    #       'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y',
    #       'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
    #       'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    #       'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    #       'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    #       'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac',
    #       'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
    #       'Fm', 'Md', 'No', 'Lr',
    #       'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut',
    #       'Fl', 'Uup', 'Lv', 'Uus']
    # for i in table1:
    #     plot_element_dependence(i)
    # plot_statistics(save=True)
    # plot_acc_on_partial_feature()
    # plot_acc_on_nspecies()
    # plot_acc_on_category()
    # plot_color_bar()
