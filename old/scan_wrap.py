import sys
#from func_scan_frequencies_for_CFC import scan_freq
#from CFC_stimulus_permutation import CFC_stimulus_permutation
#from CFC_pre_post_permutations import CFC_pre_post_permutations
#from CFC_low_high_pre_post import CFC_low_high_pre_post
from CFC_learning_stimulus import CFC_learning_stimulus

subj = 'EC77'
#block = str(sys.argv[2])
#block = ['B' + block.split('B')[1], 'B' + block.split('B')[2]]
block = 'B27'

phase_elec = int(sys.argv[1])
amp_elec = int(sys.argv[2])

surrogate_analysis = 1

print('Phase_elec ' + str(phase_elec) + ', Amp_elec ' + str(amp_elec)) 
#scan_freq(subj,block,phase_elec,amp_elec,surrogate_analysis)
#CFC_stimulus_permutation(subj,block,phase_elec,amp_elec,surrogate_analysis)
#CFC_pre_post_permutations(subj,block,phase_elec,amp_elec,surrogate_analysis)
#CFC_low_high_pre_post(subj,block,phase_elec,amp_elec,surrogate_analysis)
CFC_learning_stimulus(subj,block,phase_elec,amp_elec,'dPAC',3,surrogate_analysis)
