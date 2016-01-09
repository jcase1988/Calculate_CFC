__author__ = 'johncase'
#@profile
def calculate_CFC(data,srate,onsets,category_array,comparison_array,phase_elec,amp_elec,freqs,bandwidths,CFC_method,surrogate_analysis):

    """
    Input: 1) Data: matrix, <channels,samples>
           2) srate: integer, sampling rate
           3) onsets: numpy array, <length trials> stimulus onsets in samples
           4) category_array: numpy array, <length trials>, where values represent a group membership for a particular trial. Each group is independent of one another
           5) comparison_array: numpy array, <length trials>, where values correspond to which groups will be subtracted together, *within* each category (as specified above). (e.g., an array of [1 1 1 2 2 2] would yield a (2nd half - 1st half) comparison)
           6) phase_elec: integer, phase-providing electrode
           7) amp_elec: integer, amplitude-providing electrode
           8) freqs: numpy array, all central frequencies to be calculated
           9) bandwidths: numpy array, all bandwidths to be calculated
           10) CFC_method: string, 'MI','PAC','dPAC'
           11) surrogate_analysis: 0 or 1, compute surrogate_analysis or not

    Output: 1) CFC_groups, numpy array, <freqs,bandwidths,categories,comparisons>, CFC at all freqs and bandwidths for each category and comparison
            2) CFC_diff, numpy array, <freqs,bandwidths,categories>,   CFC at all freqs and bandwidths after comparisons within each category are subtracted
            3) CFC_diff_surrogate, numpy array, <freqs,bandwidths,categories,kruns>,   same as CFC_diff but after kruns randomized surrogate permutations
    """

    import numpy as np
    from eegfilt import eegfilt
    from scipy.signal import hilbert
    from math import log
    import pickle
    import os
    import random
    from collections import defaultdict

    #Initialize
    categories = np.unique(category_array)               #number of comparisons (e.g., number of stimIDs, values, etc.)
    comparisons = np.unique(comparison_array)            #number of groups within each comparison (e.g., 2nd half vs 1st half)

    #Number of surrogate runs
    kruns = 200

    CFC_groups = np.zeros((len(freqs),len(bandwidths),len(categories),len(comparisons))) # <frequencies,bandwidths,comparisons,categories in each comparison>
    CFC_diff = np.zeros((len(freqs),len(bandwidths),len(categories)))                     # <frequencies,bandwidths,comparisons>
    CFC_surrogate_diff = np.zeros((len(freqs),len(bandwidths),len(categories),kruns))      # <frequencies,bandwidths,comparisons,surrogate runs>

    #Amplitude-providing frequency - High Gamma
    freq_amp = np.array([70,150])

    #Trial window
    post_secs = 1  #How many seconds after onset should be analyzed

    #Keep data for only the amplitude- and phase-providing electrodes
    amp_elec_data = data[amp_elec-1,:]
    phase_elec_data = data[phase_elec-1,:]
    del data

    #Filter and calculate power for amplitude-providing electrode
    pow,filtwt = eegfilt(amp_elec_data,srate,[],freq_amp[1])  #low-pass filter
    pow,filtwt = eegfilt(pow[0],srate,freq_amp[0],[])         #high-pass filter
    pow = abs(hilbert(pow[0][0:len(pow[0])-1]))         #analytic amplitude via hilbert

    #For each trial, a tuple containing the 0) trial index, 1) trial category, 2) trial comparison, 3) trial samples, 4) trial analytic amp
    trial_data = []
    for iTrial,onset in enumerate(onsets):
        trl_category = category_array[iTrial]
        trl_comparison = comparison_array[iTrial]
        trl_samples = np.arange(onset,onset+np.rint(post_secs*srate)).astype(int)
        trial_data.extend([(iTrial,trl_category,trl_comparison,trl_samples,pow[trl_samples])])

    #Calculate MI for each phase-providing central-frequencies / bandwidths
    for iFreq,freq in enumerate(freqs):
        for iBand,band in enumerate(bandwidths):

            #Skip if bandwidth is too big for frequency
            if freq-(band/2) < 0.5:
                CFC_diff[iFreq,iBand,:] = 0
                continue

            print('freq = ' + str(freq) + ', bw = ' + str(band))

            #Filter and calculate instantaneous phase for phase-providing electrode
            pha = np.zeros([1,len(pow)])               # create time series of zeros (effectively deleting previous time series, if it exists)
            pha = eegfilt(phase_elec_data,srate,[],freq+(band/2))[0][0] # low-pass filter from 0 hz to the current freq plus half the bandwidth
            pha = eegfilt(pha,srate,freq-(band/2),[])[0][0]          # high-pass filter from current freq minus the other half the bandwidth to infinity
            pha = np.angle(hilbert(pha[0:len(pha)-1]))             # calculate instantaneous phase via hilbert

            #Update "trial_data" with phase angles for each trial
            for iTrial in range(len(trial_data)):
                trial_data[iTrial] = trial_data[iTrial][:5] + (pha[trial_data[iTrial][3]],)

            #Calculate CFC for each category/comparison combination
            for iCategory,category in enumerate(categories):
                for iComparison,comparison in enumerate(comparisons):

                    #retrieve power from trial_data and unfold power data into one list
                    pow_data = [trial[4] for trial in trial_data if trial[1] == category and trial[2] == comparison] #pick out power data for appropriate trials
                    pow_data = np.array([item for sublist in pow_data for item in sublist])                             #flatten out nested trial structure in the array, returning one, unnested list

                    #retrieve power from trial_data and unfold power data into one list
                    pha_data = [trial[5] for trial in trial_data if trial[1] == category and trial[2] == comparison] #pick out power data for appropriate trials
                    pha_data = np.array([item for sublist in pha_data for item in sublist])                             #flatten out nested trial structure in the array, returning one, unnested list


                    #http://jn.physiology.org/content/104/2/1195
                    if CFC_method == 'MI':

                        #Define phase bins
                        n_bins = 18
                        bin_size = 2*np.pi/n_bins
                        bins = np.arange(-np.pi,np.pi-bin_size,bin_size)

                        #Calculate mean amplitude within each phase bin to yield a
                        #distribution of amplitude(phase)
                        bin_dist = np.zeros([len(bins)])
                        for iBin in range(len(bins)):
                            ind = np.logical_and(pha_data>=bins[iBin],pha_data<bins[iBin]+bin_size)
                            bin_dist[iBin] = np.mean(pow_data[ind])

                        #Normalize distribution to yield pseudo probability density function
                        bin_dist = bin_dist / sum(bin_dist)

                        #Calculate Shannon entropy of PDF
                        h_p = 0
                        for iBin,mybin in enumerate(bin_dist):
                            h_p = h_p - mybin * np.log(mybin)

                        #MI = (Kullback-Leibler distance between h_p and uniform
                        #distribution) / (Entropy of uniform distribution)
                        CFC_groups[iFreq,iBand,iCategory,iComparison] = (np.log(len(bins)) - h_p) / log(len(bins))

                    elif CFC_method == 'PAC':
                        CFC_groups[iFreq,iBand,iCategory,iComparison] = np.abs(np.mean(pow_data*(np.exp(1j*pha_data))))

                    elif CFC_method == 'dPAC':
                        CFC_groups[iFreq,iBand,iCategory,iComparison] = np.abs(np.mean(pow_data*(np.exp(1j*pha_data) - np.mean(np.exp(1j*pha_data)))))

                #If two comparisons (e.g., late vs early), calculate difference.
                #Positive difference = 2 > 1, negative difference = 1 > 2
                if len(comparisons)==2:
                    CFC_diff[iFreq,iBand,iCategory] = CFC_groups[iFreq,iBand,iCategory,1] - CFC_groups[iFreq,iBand,iCategory,0]

                #If three comparisons, calculate the sum of the absolute differences between each pair.
                #High difference signifies selectivity, low difference signifies similiarity
                elif len(comparisons)==3:
                    CFC_diff[iFreq,iBand,iCategory] = np.abs(CFC_groups[iFreq,iBand,iCategory,1] - CFC_groups[iFreq,iBand,iCategory,0]) + np.abs(CFC_groups[iFreq,iBand,iCategory,2] - CFC_groups[iFreq,iBand,iCategory,0]) + np.abs(CFC_groups[iFreq,iBand,iCategory,2] - CFC_groups[iFreq,iBand,iCategory,1])

                if surrogate_analysis == 1:

                    #Calculate surrogate distribution for each category combination
                    for iCategory,category in enumerate(categories):
                        pha_trials = [trial[5] for trial in trial_data if trial[1] == category]
                        amp_trials = [trial[4] for trial in trial_data if trial[1] == category]

                        #it's not necessary to shuffle phases, so calculate them before randomization for efficiency
                        pha_data = [None]*len(comparisons)
                        cnt = 0
                        comparison_array_for_category = [trial[2] for trial in trial_data if trial[1] == category] #same as "comparison_array" but only containing elements equal to current category
                        for iComparison,comparison in enumerate(comparisons):

                            #find n-trials, where n = the number of trials in each comparison category
                            phase_for_comparison = pha_trials[cnt:cnt+sum(comparison_array_for_category==comparison)]

                            #update starting position for the previous line
                            cnt = cnt + sum(comparison_array_for_category==comparison)-1

                            #flatten out nested trial structure in the array, returning one, unnested list
                            pha_data[iComparison] = np.array([item for sublist in phase_for_comparison for item in sublist])

                        #Shuffle the amplitude time course between trials, while maintaining the structure within each time course
                        for iRun in range(kruns):

                            CFC_group_surrogate = np.zeros(len(comparisons))

                            #Shuffle amplitudes
                            random.shuffle(amp_trials)

                            cnt = 0
                            for iComparison,comparison in enumerate(comparisons):

                                amp_for_comparison = amp_trials[cnt:cnt+sum(comparison_array_for_category==comparison)]
                                cnt = cnt + sum(comparison_array_for_category==comparison)-1
                                pow_data = np.array([item for sublist in amp_for_comparison for item in sublist])

                                #http://jn.physiology.org/content/104/2/1195
                                if CFC_method == 'MI':

                                    #Calculate mean amplitude within each phase bin to yield a
                                    #distribution of amplitude(phase)
                                    bin_dist = np.zeros([len(bins)])
                                    for iBin in range(len(bins)):
                                        ind = np.logical_and(pha_data[iComparison]>=bins[iBin],pha_data[iComparison]<bins[iBin]+bin_size)
                                        bin_dist[iBin] = np.mean(pow_data)

                                    #Normalize distribution to yield pseudo probability density function
                                    bin_dist = bin_dist / sum(bin_dist)

                                    #Calculate Shannon entropy of PDF
                                    h_p = 0
                                    for iBin,mybin in enumerate(bin_dist):
                                        h_p = h_p - mybin * np.log(mybin)

                                    #MI = (Kullback-Leibler distance between h_p and uniform
                                    #distribution) / (Entropy of uniform distribution)
                                    CFC_group_surrogate[iComparison] = (np.log(len(bins)) - h_p) / log(len(bins))

                                elif CFC_method == 'PAC':
                                    CFC_group_surrogate[iComparison] = np.abs(np.mean(pow_data*(np.exp(1j*pha_data[iComparison]))))

                                elif CFC_method == 'dPAC':
                                    CFC_group_surrogate[iComparison] = np.abs(np.mean(pow_data*(np.exp(1j*pha_data[iComparison]) - np.mean(np.exp(1j*pha_data[iComparison])))))

                            #If two comparisons (e.g., late vs early), calculate difference.
                            #Positive difference = 2 > 1, negative difference = 1 > 2
                            if len(comparisons)==2:
                                CFC_surrogate_diff[iFreq,iBand,iCategory,iRun] = CFC_group_surrogate[1] - CFC_group_surrogate[0]

                            #If three comparisons, calculate the sum of the absolute differences between each pair.
                            #High difference signifies selectivity, low difference signifies similiarity
                            elif len(comparisons)==3:
                                CFC_surrogate_diff[iFreq,iBand,iCategory,iRun] = np.abs(CFC_group_surrogate[1] - CFC_group_surrogate[0]) + np.abs(CFC_group_surrogate[2] - CFC_group_surrogate[0]) + np.abs(CFC_group_surrogate[2] - CFC_group_surrogate[1])




    return CFC_groups, CFC_diff, CFC_surrogate_diff

if __name__ == "__main__":

    import scipy.io as sio
    import numpy as np

    subj = 'EC77'
    blk = 'B26'

    data_path = '/Users/johncase/Documents/UCSF Data/' + subj + '/' + subj + blk + '/data/' + subj + '_' + blk + '_data.mat'
    ecog_data = sio.loadmat(data_path)['ecogData']

    subglo_path = '/Users/johncase/Documents/UCSF Data/subj_globals.mat'
    subj_data = sio.loadmat(subglo_path,struct_as_record=False, squeeze_me=True)['subj_globals']
    subj_data = getattr(getattr(subj_data,subj),blk)

    srate = subj_data.srate
    all_onsets = subj_data.allstimtimes[:,0]
    stimID = subj_data.stimID
    values = subj_data.values
    all_onsets = np.delete(all_onsets,np.where(stimID==10),0) #delete clicks

    category_array = np.array(values[stimID-1])
    comparison_array = np.array([0 if index<len(stimID)/2 else 1 for index in np.arange(len(stimID))])

    freqs = np.arange(1,21)
    bandwidths = np.arange(1,11)

    phase_elec = 10
    amp_elec = 10
    output_path = '/Users/johncase/Documents/test/e' + str(phase_elec) + '_e' + str(amp_elec) + '_'

    CFC_groups,CFC_diff,CFC_surrogate_diff = calculate_within_CFC(ecog_data,srate,all_onsets,category_array,comparison_array,phase_elec,amp_elec,freqs,bandwidths,'dPAC',1)

    np.save(open(output_path+'diff_surrogate','wb'),CFC_surrogate_diff)
    np.save(open(output_path+'diff','wb'),CFC_diff)
    np.save(open(output_path+'groups','wb'),CFC_groups)
