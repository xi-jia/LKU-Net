import numpy as np
import csv, sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import wilcoxon, ttest_rel, ttest_ind

outstruct = ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebral-White-Matter', 'Cerebellum-White-Matter', 'Putamen', 'VentralDC', 'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
             '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF', 'choroid-plexus']
exp_data = np.zeros((len(outstruct), 115))
stct_i = 0
file_dir = './Quantitative_Results/'
for stct in outstruct:
    tar_idx = []
    with open(file_dir+'L2ss_2_Chan_8_LR_0.0001_Smooth_5.0_Test.csv', "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            if i == 1:
                names = line[0].split(',')
                idx = 0
                for item in names:
                    if stct in item:
                        tar_idx.append(idx)
                    idx += 1
            elif i>1:
                if line[0].split(',')[1]=='':
                    continue
                val = 0
                for lr_i in tar_idx:
                    vals = line[0].split(',')
                    val += float(vals[lr_i])
                val = val/len(tar_idx)
                exp_data[stct_i, i-2] = val
    stct_i+=1
# all_dsc.append(exp_data.mean(axis=0))
print(exp_data.mean())
print(exp_data.std())
my_list = []
with open(file_dir+'L2ss_2_Chan_8_LR_0.0001_Smooth_5.0_Test.csv', newline='') as f:
    reader = csv.reader(f)
    my_list = [row[-1] for row in reader]
my_list = my_list[2:]
my_list = np.array([float(i) for i in my_list])*100
print('jec_det: {:.3f} +- {:.3f}'.format(my_list.mean(), my_list.std()))
