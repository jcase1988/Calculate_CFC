1. Running PAC from server

Access my account via:

ssh -p 7777 jcase@dura.cin.ucsf.edu

CFC_wrapper.py
Directory ~/code/ contains the script “CFC_wrapper.py”. This script takes care of setting up parameters for “calculate_CFC.py” (e.g., which frequencies, bandwidths to use), handles data, and defines the category and comparison arrays.

You will want to change at least three variables:

1) Category array is an array of indicator variables which indicate trials that are independent of each other. For example, to do one CFC calculation for each stimulus_ID, set category_array = StimID.

2) Comparison array is an array of indicator variables which indicate which trials will be compared within each category. For example, to do 2nd half vs 1st half, set comparison_array = [0,0,0,…,1,1,1].

3) You will want to change “output_path” to the desired directory you wish to save the CFC calculations.

“CFC_wrapper.py” will then run “calculate_CFC.py”, which will output three variables to output_path:

1) e(phase_elec)_e(amp_elec)_groups: <frequencies, bandwidths, categories, comparisons>

2) e(phase_elec)_e(amp_elec)_diff: <frequencies, bandwidths, categories> (if two comparisons, e(phase_elec)_e(amp_elecs)_groups[:,:,:,1] - e(phase_elec)_e(amp_elecs)_groups[:,:,:,0])

3) e(phase_elec)_e(amp_elecs)_surrogate_diff <frequencies, bandwidths,categories,runs>


CFC_wrapper.py can be run in the command line:

python /home/jcase/code/CFC_wrapper.py SUBJ BLOCK PHASE_ELEC AMP_ELEC &

For example, if you are running only one block:
python /home/jcase/code/CFC_wrapper.py EC77 B26 100 100 &

Or if you are using two blocks:
python /home/jcase/code/CFC_wrapper.py EC77 [B26,B42] 100 100 &

doCFC_ALL:
If you would like to run CFC_wrapper more systematically, the bash script “/home/jcase/doCFC_all” will iterate through all subjects, blocks, and electrodes (running 4 CFC_wrapper jobs in parallel, one for each electrode pair). If you would like to adjust the electrodes run, you will need to change the “elec_start” and “elec_end” arrays. If you would like to adjust the subjects and blocks run, change the “subj” and “block” arrays.

If you need to kill this process, try “killall bash” and then “killall python”.
