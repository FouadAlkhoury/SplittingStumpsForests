# semiquantitative scoring of histopathological assessment
import matplotlib.pyplot as plt
import numpy as np
import csv

resultsPath = "../tmp/reports/"

groups = ['Control']
#groups = ['Control','dic','se','dic + se']
dilatation_control = [0,1,1,1,2,1,1,2]
dilatation_dic = [3,3,3,3,3,3,3,3]
dilatation_se = [2,1,1,0,2,1,1,2]
dilatation_dic_se = [3,3,2,3,3,3,3]
mean_di_cont = np.mean(dilatation_control)
mean_di_dic = np.mean(dilatation_dic)
mean_di_se = np.mean(dilatation_se)
mean_di_dic_se = np.mean(dilatation_dic_se)
se_di_cont = np.std(dilatation_control,ddof = 1)/np.sqrt(8)
se_di_dic = np.std(dilatation_dic,ddof = 1)/np.sqrt(8)
se_di_se = np.std(dilatation_se,ddof = 1)/np.sqrt(8)
se_di_dic_se = np.std(dilatation_dic_se,ddof = 1)/np.sqrt(7)
print('Dilatation')
print(mean_di_cont)
print(se_di_cont)
print(mean_di_dic)
print(se_di_dic)
print(mean_di_se)
print(se_di_se)
print(mean_di_dic_se)
print(se_di_dic_se)

vacuolation_control = [1,1,2,1,1,0,0,1]
vacuolation_dic = [3,3,3,2,3,3,3,3]
vacuolation_se = [1,2,0,1,1,0,2,1]
vacuolation_dic_se = [3,2,2,2,2,3,2]
mean_va_cont = np.mean(vacuolation_control)
mean_va_dic = np.mean(vacuolation_dic)
mean_va_se = np.mean(vacuolation_se)
mean_va_dic_se = np.mean(vacuolation_dic_se)
se_va_cont = np.std(vacuolation_control,ddof = 1)/np.sqrt(8)
se_va_dic = np.std(vacuolation_dic,ddof = 1)/np.sqrt(8)
se_va_se = np.std(vacuolation_se,ddof = 1)/np.sqrt(8)
se_va_dic_se = np.std(vacuolation_dic_se,ddof = 1)/np.sqrt(7)


print('vacuolation')
print(mean_va_cont)
print(se_va_cont)
print(mean_va_dic)
print(se_va_dic)
print(mean_va_se)
print(se_va_se)
print(mean_va_dic_se)
print(se_va_dic_se)

dy_control = [0,0,0,0,0,1,1,1]
dy_dic = [3,2,2,3,3,3,2,1]
dy_se = [2,2,0,0,1,0,0,0]
dy_dic_se = [2,3,3,1,2,2,2]
mean_dy_cont = np.mean(dy_control)
mean_dy_dic = np.mean(dy_dic)
mean_dy_se = np.mean(dy_se)
mean_dy_dic_se = np.mean(dy_dic_se)
se_dy_cont = np.std(dy_control,ddof = 1)/np.sqrt(8)
se_dy_dic = np.std(dy_dic,ddof = 1)/np.sqrt(8)
se_dy_se = np.std(dy_se,ddof = 1)/np.sqrt(8)
se_dy_dic_se = np.std(dy_dic_se,ddof = 1)/np.sqrt(8)


print('nicrosis')
print(mean_dy_cont)
print(se_dy_cont)
print(mean_dy_dic)
print(se_dy_dic)
print(mean_dy_se)
print(se_dy_se)
print(mean_dy_dic_se)
print(se_dy_dic_se)


kb_control = [0,1,0,1,0,0,1,0]
kb_dic = [2,2,2,2,2,2,2,2]
kb_se = [0,1,1,0,0,1,1,0]
kb_dic_se = [0,2,2,2,2,2,2]
mean_kb_cont = np.mean(kb_control)
mean_kb_dic = np.mean(kb_dic)
mean_kb_se = np.mean(kb_se)
mean_kb_dic_se = np.mean(kb_dic_se)
se_kb_cont = np.std(kb_control,ddof = 1)/np.sqrt(8)
se_kb_dic = np.std(kb_dic,ddof = 1)/np.sqrt(8)
se_kb_se = np.std(kb_se,ddof = 1)/np.sqrt(8)
se_kb_dic_se = np.std(kb_dic_se,ddof = 1)/np.sqrt(7)


di = [mean_di_cont,mean_di_dic,mean_di_se,mean_di_dic_se]
sem = [se_di_cont,se_di_dic,se_di_se,se_di_dic_se]
width = 0.16
space = 0.1
x_pos = np.arange(len(groups))
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(x_pos-width,mean_di_cont,width,yerr=se_di_cont,color='b',label='Tubular Dilatation', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+5*width,mean_di_dic,width,yerr=se_di_dic,color='b', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+11*width,mean_di_se,width,yerr=se_di_se,color='b', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+17*width,mean_di_dic_se,width,yerr=se_di_dic_se,color='b', align='center', alpha=0.5, ecolor='black', capsize=10)

ax.bar(x_pos,mean_va_cont,width,yerr=se_va_cont,color='r', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+6*width,mean_va_dic,width,yerr=se_va_dic,color='r',label='Tubular Vacuolation', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+12*width,mean_va_se,width,yerr=se_va_se,color='r', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+18*width,mean_va_dic_se,width,yerr=se_va_dic_se,color='r', align='center', alpha=0.5, ecolor='black', capsize=10)

ax.bar(x_pos + width,mean_dy_cont,width,yerr=se_dy_cont,color='g', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+7*width,mean_dy_dic,width,yerr=se_dy_dic,color='g', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+13*width,mean_dy_se,width,yerr=se_dy_se,color='g',label='Tubular Degeneration', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+19*width,mean_dy_dic_se,width,yerr=se_dy_dic_se,color='g', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos + 2 * width,mean_kb_cont,width,yerr=se_kb_cont,color='brown', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+8*width,mean_kb_dic,width,yerr=se_kb_dic,color='brown', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+14*width,mean_kb_se,width,yerr=se_kb_se,color='brown', align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_pos+20*width,mean_kb_dic_se,width,yerr=se_kb_dic_se,color='brown',label='Glomerular Pathology', align='center', alpha=0.5, ecolor='black', capsize=10)

x = ['Control','Dic','Se','Dic + Se','']
y = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

default_x_ticks = range(len(x))
ax.set_xticks(default_x_ticks)
ax.set_xticklabels(x)
ax.set_yticks(y)
ax.set_ylabel('Injury Degree')
plt.legend(bbox_to_anchor=(0.3, -0.1), loc='upper left')
plt.text(x_pos+4.5*width,3.1,'a')
plt.text(x_pos+5.7*width,3.1,'a')
plt.text(x_pos+6.9*width,2.7,'a')
plt.text(x_pos+7.9*width,2.1,'a')
plt.text(x_pos+17.7*width,2.5,'*')


fig = plt.gcf()
fig.savefig(resultsPath+'groups.png', bbox_inches="tight", dpi=650)
fig.show()


