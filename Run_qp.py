'''

  _____                   _                ____  _____              _                
 |  __ \                 (_)              / __ \|  __ \            | |               
 | |__) |   _ _ __  _ __  _ _ __   __ _  | |  | | |__) |  ___  ___ | |_   _____ _ __ 
 |  _  / | | | '_ \| '_ \| | '_ \ / _` | | |  | |  ___/  / __|/ _ \| \ \ / / _ \ '__|
 | | \ \ |_| | | | | | | | | | | | (_| | | |__| | |      \__ \ (_) | |\ V /  __/ |   
 |_|  \_\__,_|_| |_|_| |_|_|_| |_|\__, |  \___\_\_|      |___/\___/|_| \_/ \___|_|   
                                   __/ |                                             
                                  |___/                                              

'''
print(__doc__)

from cut_select_qpM_kmeans import CutSolverK # import solver class
import numpy as np 
import pandas as pd
import itertools
import os
from timeit import default_timer as timer 

# Define paremeters
sel_size=100 # number of selected cuts
dim=3 # dimension of low dimensional cuts, only triplets used
cut_rounds=20 # iterations
termon= False
boxqpinst = ["spar070-050-1"] # name of file to input Q
# "spar100-025-1", "spar070-050-1","spar070-075-1","spar080-025-1", "spar080-050-1","spar080-075-1","spar090-025-1", "spar090-050-1","spar090-075-1","spar100-025-1", "spar100-050-1","spar100-075-1","spar125-025-1","spar125-050-1","spar125-075-1"] 

# Initiallize
csK=CutSolverK()

strat = 1  #1= feasibility, 2= optimality, 4=combined, 5= random
strategies = {1: 'feasibility', 2: 'optimality'}

for filename in boxqpinst : # iterate over boxqp instances
  for n_clusters in  [100]: 
      print('The filename is', filename, 'The strategy is', strat)
      print('The number of clusters is', n_clusters)
      (solK, timeK, round_times, sep_times, nbs_sdp_cutsK, nbs_tri_cuts, vars_values, agg_list) = \
       csK.cut_select_algo(filename=filename, dim=dim, sel_size=sel_size, \
        strat=strat, nb_rounds_cuts=cut_rounds,term_on=termon)
      print ('Final solution is', solK[-1], 'The new time elapsed is', timeK, 'The cut rounds are',len(nbs_sdp_cutsK)-1)
  
# save to csv file at the end
results={'Solution': solK,'Time': timeK}
df = pd.DataFrame(results)
# Change the current working directory to the folder you wanna save the .csv file
os.chdir('C:/Users/mexis/OneDrive/Υπολογιστής/SDPCutSel/Results/')
File_name='Kmeans ' + strategies[strat] + ' ' + str(n_clusters) + ' clusters.csv'
df.to_csv(File_name, header='column_names',index=False)
