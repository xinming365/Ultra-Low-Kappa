auid: aflow下载的数据列表。5603个。
labels_2:与new_auid相对应的label。形状为(5603,5)
train_data_2:与new_auid相对应的原始训练数据。形状为(5603,16)
positions_fractional:共5603个。但是其中存在117个数据是空的。
最后剩下数据个数：5486.
labels文件，标签分别是：

agl_thermal_conductivity_300K 热导率
agl_debye 德拜温度
agl_heat_capacity_Cp_300K 定压热容
agl_heat_capacity_Cv_300K 定容热容
agl_thermal_expansion_300K 热膨胀系数

train_data.npy: 最原始的下载数据，shape(5486,16)。aurl/auid/compound/(a,b,c)/(alpha,beta,gamma)/density/volume_cell/volume per atom(volume_atom);number of atoms in unit cell(natoms);nspecies;晶系;空间群

train_version_1: 数据的shape为(5486,56)
6(晶体直接相关性质：空间群/原子种类数/原子数/单位原子体积/单位原胞体积/密度)+25(根据原子性质求和构建)+25(晶体结构构建)
train_version_2: 数据的shape为(5486,61)
在1的基础上加入了整体的统计量：5个性质(mean,min,max,std,range)
train_version_3: 数据的shape为(5486,8,8,1)
在1的基础上：1(cwZ)+7(晶系+...)+25+25+6(统计量)
train_version_4: 结构数据。
train_version_5: 最终数据，shape(5486,75).晶体性质7(晶系/空间群/原子种类数/原子数/单位原子体积/单位原胞体积/密度(来自train_version_1))+ 组分权重性质25+ 晶体结构指纹25+ 统计量18(6晶体指纹+6距离矩阵+6邻接矩阵)
train_version_6: 热导率数据库（输入）
labels_6:热导率数据库（热导）（log形式）
