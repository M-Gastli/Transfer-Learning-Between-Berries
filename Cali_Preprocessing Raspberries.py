#%%
import matplotlib.pyplot as plt
import numpy as np
#%%

'''
    Lagging Histograms with Old Yield
'''
import pandas as pd
import numpy as np

final_hists = np.load('final_hists.npy')
final_hists = np.transpose(final_hists, (0,2,1))

yld = pd.read_excel('Raspberries.xlsx')
print(final_hists.shape, yld.shape)

# Aligning histogram dates with available yield
#final_hists = np.delete(final_hists, np.arange(31+28+31+30+31+30+31+31+24 - 140 - 35), axis=0)
print(final_hists.shape)
#%%

yields = yld['Imputed Pounds/Acre'].values

lagged_hists = []
a = 35
for i in range(len(final_hists)):
    if i + 140 + a >= len(final_hists):
        break
    lagged_hists.append([final_hists[i:i+140], yields[i+140+a]])
lagged_hists = np.array(lagged_hists)
print(lagged_hists.shape)
np.save('lagged_hists_35rasp', lagged_hists, allow_pickle=True)