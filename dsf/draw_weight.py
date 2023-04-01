# kidney weight / body weight ratio
import matplotlib.pyplot as plt
import numpy as np
import csv


resultsPath = "../tmp/reports/"

weight_control = [0.6158,0.5285,0.5508,0.5622,0.5218,0.5657,0.5135,0.5652]
weight_dic = [0.8044,0.6178,0.6651,0.6095,0.6028,0.7202,0.8066,0.9939]
weight_se = [0.5733,0.591,0.6044,0.5433,0.5363,0.5168,0.5026,0.5891]
weight_dic_se = [0.5995,0.6456,0.6576,0.813,0.742,0.8601,0.7897]

groups = ['Control']
groups = ['Control','dic','se','dic + se']
dilatation_control = [0,1,1,1,2,1,1,2]
dilatation_dic = [3,3,3,3,3,3,3,3]
dilatation_se = [2,1,1,0,2,1,1,2]
dilatation_dic_se = [3,3,2,3,3,3,3]

mean_weight_cont = np.mean(weight_control)
mean_weight_dic = np.mean(weight_dic)
mean_weight_se = np.mean(weight_se)
mean_weight_dic_se = np.mean(weight_dic_se)

se_weight_cont = np.std(weight_control,ddof = 1)/np.sqrt(8)
se_weight_dic = np.std(weight_dic,ddof = 1)/np.sqrt(8)
se_weight_se = np.std(weight_se,ddof = 1)/np.sqrt(8)
se_weight_dic_se = np.std(weight_dic_se,ddof = 1)/np.sqrt(7)

mean = [mean_weight_cont,mean_weight_dic,mean_weight_se,mean_weight_dic_se]
se = [se_weight_cont,se_weight_dic,se_weight_se,se_weight_dic_se]
width = 0.3
space = 0.1
x_pos = np.arange(len(groups))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x_pos,mean,width,yerr=se,color='b', align='center', alpha=0.6, ecolor='black', capsize=10)

x = ['Control','Dic','Se','Dic + Se']
y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

default_x_ticks = range(len(x))
ax.set_xticks(default_x_ticks)
ax.set_xticklabels(x)
ax.set_yticks(y)
ax.set_xlabel('Groups')
ax.set_ylabel('Kidney Weight / Body Weight (%)')
plt.text(0.95,0.8,'a')
fig = plt.gcf()
fig.savefig(resultsPath+'weight.png', bbox_inches="tight", dpi=650)
fig.show()


