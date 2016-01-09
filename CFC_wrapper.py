import sys
import scipy.io as sio
import numpy as np
from calculate_CFC import calculate_CFC

subj = str(sys.argv[1])
#if mutliple blocks
if ',' in sys.argv[2]:
    block = sys.argv[2].strip('[').strip(']').split(',')
    phase_elec = int(sys.argv[3])
    amp_elec = int(sys.argv[4])

    surrogate_analysis = 1
    CFC_method = 'dPAC'

    print('Phase_elec ' + str(phase_elec) + ', Amp_elec ' + str(amp_elec))

    #Paths
    subglo_path = '/home/jcase/data/subj_globals'

    output_path = '/home/jcase/data/' + subj + ''.join(block) + '/test/e' + str(phase_elec) + '_e' + str(amp_elec) + '_'

    ecog_data = [None] * len(block)
    all_onsets = [None] * len(block)
    stimID = [None] * len(block) 
    values = [None] * len(block)
    for iBlock,blk in enumerate(block):
        #Load and parse data
        data_path = '/home/jcase/data/' + subj + blk + '/' + subj + '_' + blk + '_data.mat'
        ecog_data[iBlock] = sio.loadmat(data_path)['ecogData']
	subj_data = sio.loadmat(subglo_path,struct_as_record=False, squeeze_me=True)['subj_globals']
        subj_data = getattr(getattr(subj_data,subj),blk)
        srate = subj_data.srate
	all_onsets[iBlock] = np.rint(subj_data.allstimtimes[:,0] * srate).astype(int)
	stimID[iBlock] = subj_data.stimID
        all_onsets[iBlock] = np.delete(all_onsets[iBlock],np.where(stimID[iBlock]==10),0) #delete clicks
        values[iBlock] = subj_data.values[stimID[iBlock]-1]
        #Adjust onsets so that they will align with data matrix after concatenation
        if iBlock > 0:
            all_onsets[iBlock] = all_onsets[iBlock] + ecog_data[iBlock-1].shape[1]

    ecog_data = np.concatenate(ecog_data,axis=1)
    all_onsets = np.concatenate(all_onsets)
    stimID = np.concatenate(stimID)
    values = np.concatenate(values)



#if only 1 block
else:
    block = str(sys.argv[2])
    phase_elec = int(sys.argv[3])
    amp_elec = int(sys.argv[4])

    surrogate_analysis = 1
    CFC_method = 'dPAC'

    print('Phase_elec ' + str(phase_elec) + ', Amp_elec ' + str(amp_elec))

    #Paths
    subglo_path = '/home/jcase/data/subj_globals'
    data_path = '/home/jcase/data/' + subj + block + '/' + subj + '_' + block + '_data.mat'
    output_path = '/home/jcase/data/' + subj + block + '/test/e' + str(phase_elec) + '_e' + str(amp_elec) + '_'

    #Load and parse aata
    ecog_data = sio.loadmat(data_path)['ecogData']
    subj_data = sio.loadmat(subglo_path,struct_as_record=False, squeeze_me=True)['subj_globals']
    subj_data = getattr(getattr(subj_data,subj),block)
    srate = subj_data.srate
    all_onsets = np.rint(subj_data.allstimtimes[:,0]*srate).astype(int)
    stimID = subj_data.stimID
    all_onsets = np.delete(all_onsets,np.where(stimID==10),0) #delete clicks
    values = subj_data.values[stimID-1]

#Make Category and Comparison Arrays
category_array = values
comparison_array = np.array([0 if index<len(stimID)/2 else 1 for index in np.arange(len(stimID))])

frequencies = np.arange(1,3)
bandwidths = np.arange(1,2)

CFC_groups,CFC_diff,CFC_surrogate_diff = calculate_CFC(ecog_data,srate,all_onsets,category_array,comparison_array,phase_elec,amp_elec,frequencies,bandwidths,CFC_method,surrogate_analysis)

np.save(open(output_path+'diff_surrogate','wb'),CFC_surrogate_diff)
np.save(open(output_path+'diff','wb'),CFC_diff)
np.save(open(output_path+'groups','wb'),CFC_groups)




