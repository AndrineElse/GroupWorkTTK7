#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:58:41 2020

@author: andrine
"""


import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import math
import scipy.io.wavfile as wf
import scipy.signal
import wavio


raw_data_path = '../../data/Kaggle/raw/'
patient_info_path = '../../data/Kaggle/external/'
data_path = '../../data/Kaggle/processed/'

def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)



def get_patient_df():
    df_no_diagnosis = pd.read_csv(patient_info_path + 'demographic_info.txt', names =
                 ['Patient number', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)'],
                 delimiter = ' ')

    diagnosis = pd.read_csv(patient_info_path + 'patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])

    df =  df_no_diagnosis.join(diagnosis.set_index('Patient number'), on = 'Patient number', how = 'left')
    print(df['Diagnosis'].value_counts())
    return df

def get_recording_info():
    root = raw_data_path
    filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
    i_list = []
    rec_annotations = []
    rec_annotations_dict = {}
    for s in filenames:
        (i,a) = Extract_Annotation_Data(s, root)
        i_list.append(i)
        rec_annotations.append(a)
        rec_annotations_dict[s] = a
    recording_info = pd.concat(i_list, axis = 0)
    print(recording_info.head())
    return rec_annotations, rec_annotations_dict


def get_file_label_df():
    _ , rec_annotations_dict = get_recording_info()
    filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
    no_label_list = []
    crack_list = []
    wheeze_list = []
    both_sym_list = []
    filename_list = []
    for f in filenames:
        d = rec_annotations_dict[f]
        no_labels = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
        n_crackles = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
        n_wheezes = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
        both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
        no_label_list.append(no_labels)
        crack_list.append(n_crackles)
        wheeze_list.append(n_wheezes)
        both_sym_list.append(both_sym)
        filename_list.append(f)
    file_label_df = pd.DataFrame(data = {'filename':filename_list, 'no label':no_label_list, 'crackles only':crack_list, 'wheezes only':wheeze_list, 'crackles and wheezees':both_sym_list})
    return file_label_df

def print_distribution_of_data():
    w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (file_label_df['wheezes only'] != 0) | (file_label_df['crackles and wheezees'] != 0)]
    print(file_label_df.sum())

#Will resample all files to the target sample rate and produce a 32bit float array
def read_wav_file(str_filename, target_rate):
    wav = wave.open(str_filename, mode = 'r')
    (sample_rate, data) = extract2FloatArr(wav,str_filename)

    if (sample_rate != target_rate):
        ( _ , data) = resample(sample_rate, data, target_rate)

    wav.close()
    return (target_rate, data.astype(np.float32))

def resample(current_rate, data, target_rate):
    x_original = np.linspace(0,100,len(data))
    x_resampled = np.linspace(0,100, int(len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))

# -> (sample_rate, data)
def extract2FloatArr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1,2,4]:
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1:255, 2:32768}
        if bps in [1,2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor)) #clamp to [0.0,1.0]
        return (rate, data)

    elif bps == 3:
        #24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception('Unrecognized wave format: {} bytes per sample'.format(bps))

#Note: This function truncates the 24 bit samples to 16 bits of precision
#Reads a wave object returned by the wave.read() method
#Returns the sample rate, as well as the audio in the form of a 32 bit float numpy array
#(sample_rate:float, audio_data: float[])
def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames,-1)
    short_output = np.empty((nFrames, 2), dtype = np.int8)
    short_output[:,:] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))  #return numpy array to save memory via array slicing

def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels()) #bytes per sample
    return (bps, lp_wave.getnchannels())

def slice_data(start, end, raw_data,  sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]


def print_distribution_resp_cycles():
    duration_list = []
    rec_annotations , _ = get_recording_info()
    print('hellow')
    for i in range(len(rec_annotations)):
        current = rec_annotations[i]
        duration = current['End'] - current['Start']
        duration_list.extend(duration)

    duration_list = np.array(duration_list)
    plt.hist(duration_list, bins = 50)
    print('longest cycle:{}'.format(max(duration_list)))
    print('shortest cycle:{}'.format(min(duration_list)))
    threshold = 5
    print('Fraction of samples less than {} seconds:{}'.format(threshold,
                                                               np.sum(duration_list < threshold)/len(duration_list)))

######################## MEL SPECTROGRAM IMPLIMENTATION (WITH VTLP) ##########################################

def sample2MelSpectrum(cycle_info, sample_rate, n_filters, vtlp_params):
    n_rows = 175 # 7500 cutoff
    n_window = 512 #~25 ms window
    (f, t, Sxx) = scipy.signal.spectrogram(cycle_info[0],fs = sample_rate, nfft= n_window, nperseg=n_window)
    Sxx = Sxx[:n_rows,:].astype(np.float32) #sift out coefficients above 7500hz, Sxx has 196 columns
    mel_log = FFT2MelSpectrogram(f[:n_rows], Sxx, sample_rate, n_filters, vtlp_params)[1]
    mel_min = np.min(mel_log)
    mel_max = np.max(mel_log)
    diff = mel_max - mel_min
    norm_mel_log = (mel_log - mel_min) / diff if (diff > 0) else np.zeros(shape = (n_filters,Sxx.shape[1]))
    if (diff == 0):
        print('Error: sample data is completely empty')
    labels = [cycle_info[1], cycle_info[2]] #crackles, wheezes flags
    return (np.reshape(norm_mel_log, (n_filters,Sxx.shape[1],1)).astype(np.float32), # 196x64x1 matrix
            label2onehot(labels))

def Freq2Mel(freq):
    return 1125 * np.log(1 + freq / 700)

def Mel2Freq(mel):
    exponents = mel / 1125
    return 700 * (np.exp(exponents) - 1)

#Tased on Jaitly & Hinton(2013)
#Takes an array of the original mel spaced frequencies and returns a warped version of them
def VTLP_shift(mel_freq, alpha, f_high, sample_rate):
    nyquist_f = sample_rate / 2
    warp_factor = min(alpha, 1)
    threshold_freq = f_high * warp_factor / alpha
    lower = mel_freq * alpha
    higher = nyquist_f - (nyquist_f - mel_freq) * ((nyquist_f - f_high * warp_factor) / (nyquist_f - f_high * (warp_factor / alpha)))

    warped_mel = np.where(mel_freq <= threshold_freq, lower, higher)
    return warped_mel.astype(np.float32)

#mel_space_freq: the mel frequencies (HZ) of the filter banks, in addition to the two maximum and minimum frequency values
#fft_bin_frequencies: the bin freqencies of the FFT output
#Generates a 2d numpy array, with each row containing each filter bank
def GenerateMelFilterBanks(mel_space_freq, fft_bin_frequencies):
    n_filters = len(mel_space_freq) - 2
    coeff = []
    #Triangular filter windows
    #ripped from http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    for mel_index in range(n_filters):
        m = int(mel_index + 1)
        filter_bank = []
        for f in fft_bin_frequencies:
            if(f < mel_space_freq[m-1]):
                hm = 0
            elif(f < mel_space_freq[m]):
                hm = (f - mel_space_freq[m-1]) / (mel_space_freq[m] - mel_space_freq[m-1])
            elif(f < mel_space_freq[m + 1]):
                hm = (mel_space_freq[m+1] - f) / (mel_space_freq[m + 1] - mel_space_freq[m])
            else:
                hm = 0
            filter_bank.append(hm)
        coeff.append(filter_bank)
    return np.array(coeff, dtype = np.float32)

#Transform spectrogram into mel spectrogram -> (frequencies, spectrum)
#vtlp_params = (alpha, f_high), vtlp will not be applied if set to None
def FFT2MelSpectrogram(f, Sxx, sample_rate, n_filterbanks, vtlp_params = None):
    (max_mel, min_mel)  = (Freq2Mel(max(f)), Freq2Mel(min(f)))
    mel_bins = np.linspace(min_mel, max_mel, num = (n_filterbanks + 2))
    #Convert mel_bins to corresponding frequencies in hz
    mel_freq = Mel2Freq(mel_bins)

    if(vtlp_params is None):
        filter_banks = GenerateMelFilterBanks(mel_freq, f)
    else:
        #Apply VTLP
        (alpha, f_high) = vtlp_params
        warped_mel = VTLP_shift(mel_freq, alpha, f_high, sample_rate)
        filter_banks = GenerateMelFilterBanks(warped_mel, f)

    mel_spectrum = np.matmul(filter_banks, Sxx)
    return (mel_freq[1:-1], np.log10(mel_spectrum  + float(10e-12)))

#labels proved too difficult to train (model keep convergining to statistical mean)
#Flattened to onehot labels since the number of combinations is very low
def label2onehot(c_w_flags):
    c = c_w_flags[0]
    w = c_w_flags[1]
    if((c == False) & (w == False)):
        return [1,0,0,0]
    elif((c == True) & (w == False)):
        return [0,1,0,0]
    elif((c == False) & (w == True)):
        return [0,0,1,0]
    else:
        return [0,0,0,1]

######################## DATA PREP UTILITY FUNCTIONS ##########################################
#Used to split each individual sound file into separate sound clips containing one respiratory cycle each
#output: [filename, (sample_data:np.array, start:float, end:float, crackles:bool(float), wheezes:bool(float)) (...) ]
def get_sound_samples(recording_annotations, file_name, root, sample_rate):
    sample_data = [file_name]
    (rate, data) = read_wav_file(os.path.join(root, file_name + '.wav'), sample_rate)

    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['Start']
        end = row['End']
        crackles = row['Crackles']
        wheezes = row['Wheezes']
        audio_chunk = slice_data(start, end, data, rate)
        sample_data.append((audio_chunk, start,end,crackles,wheezes))
    return sample_data

#Fits each respiratory cycle into a fixed length audio clip, splits may be performed and zero padding is added if necessary
#original:(arr,c,w) -> output:[(arr,c,w),(arr,c,w)]
def split_and_pad(original, desiredLength, sampleRate):
    output_buffer_length = int(desiredLength * sampleRate)
    soundclip = original[0]
    n_samples = len(soundclip)
    total_length = n_samples / sampleRate #length of cycle in seconds
    n_slices = int(math.ceil(total_length / desiredLength)) #get the minimum number of slices needed
    samples_per_slice = n_samples // n_slices
    src_start = 0 #Staring index of the samples to copy from the original buffer
    output = [] #Holds the resultant slices
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start
        copy = generate_padded_samples(soundclip[src_start:src_end], output_buffer_length)
        output.append((copy, original[1], original[2]))
        src_start += length
    return output

def generate_padded_samples(source, output_length):
    copy = np.zeros(output_length, dtype = np.float32)
    src_length = len(source)
    frac = src_length / output_length
    if(frac < 0.5):
        #tile forward sounds to fill empty space
        cursor = 0
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        copy[:src_length] = source[:]
    #
    return copy

def extract_all_training_samples_alt(filenames, annotation_dict, root, target_rate):
    cycle_dict = {}
    for file in filenames:
        data = get_sound_samples(annotation_dict[file], file, root, target_rate)
        cycles_with_labels = [(d[0], d[3], d[4]) for d in data[1:]]
        cycle_dict[file] = cycles_with_labels
    return cycle_dict

def create_wav_split_files(target_sample_rate):
    filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
    root = raw_data_path
    _ , rec_annotations_dict = get_recording_info()
    sample_dict = extract_all_training_samples_alt(filenames, rec_annotations_dict, root, target_sample_rate)

    for key in sample_dict:
        i = 0
        for audio in sample_dict[key]:
            folder = ''
            if ((audio[1] == 1) and (audio[2] == 1)):
                folder = 'both/'
            elif (audio[1] == 1):
                folder = 'crackle/'
            elif (audio[2] == 1):
                folder = 'wheeze/'
            else:
                folder = 'none/'
            wavio.write(path + folder + key +'_'+ str(i) + '.wav',  audio[0], fs, sampwidth=3)
            i = 1 + i


def create_wav_split_files_same_len(sample_length_seconds , target_sample_rate, path):
    filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
    root = raw_data_path
    _ , rec_annotations_dict = get_recording_info()
    sample_dict = extract_all_training_samples_alt(filenames, rec_annotations_dict, root, target_sample_rate)

    for key in sample_dict:
        i = 0
        for audio in sample_dict[key]:
            folder = ''
            audio_padded = split_and_pad(audio, sample_length_seconds , target_sample_rate)
            for slice in audio_padded:
                if ((audio[1] == 1) and (audio[2] == 1)):
                    folder = 'both/'
                elif (audio[1] == 1):
                    folder = 'crackle/'
                elif (audio[2] == 1):
                    folder = 'wheeze/'
                else:
                    folder = 'none/'
                wavio.write(path + folder + key +'_'+ str(i) + '.wav',  slice[0], fs, sampwidth=3)

                i = 1 + i

################ SOME HELPING FUNCTIONS FOR PLOTTING ######################################


def get_diagnosis_df():
    '''
    Returns pandas dataframe with the PersonID and corresponding diagnose, list of all the types of diseases
    '''
    diagnosis_df = pd.read_csv(patient_info_path + 'patient_diagnosis.csv', sep=",", names=['pId', 'diagnosis'])
    ds = diagnosis_df['diagnosis'].unique()
    return diagnosis_df, ds

def get_filename_info(filename):
    return filename.split('_')

def get_file_info_df():
    file_names = [s.split('.')[0] for s in os.listdir(path = raw_data_path) if '.txt' in s]
    file_paths = [os.path.join(raw_data_path, file_name) for file_name in file_names]

    files_ = []
    for f in file_names:
        df = pd.read_csv(raw_data_path + '/' + f + '.txt', sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
        df['filename'] = f
        #get filename features
        f_features = get_filename_info(f)
        df['pId'] = f_features[0]
        df['ac_mode'] = f_features[3]

        files_.append(df)

    files_df = pd.concat(files_)
    files_df.reset_index()
    files_df.head()
    return files_df

def get_complete_df(target = 'wheeze/crackle'):
    '''
    Returns
    -------
    dataframe with file info, diagnosis of patient

    '''
    diagnosis_df,_ = get_diagnosis_df()
    file_info_df = get_file_info_df()


    file_info_df['pId'] = file_info_df['pId'].astype('int64')
    df = pd.merge(file_info_df, diagnosis_df, on='pId')

    df = df.reset_index()
    df = set_target_of_df(df, target = target)
    i = 0
    new_name = []
    for idx , row in df.iterrows():
        # Idx is the index of the row, and row contains all the data at the given intex
        f = row['filename']

        if idx != 0:
            if df.iloc[idx - 1]['filename'] == f:
                i = i + 1
            else:
                i = 0
        sliced_file_name = f + '_' + str(i) + '.wav'
        new_name.append(sliced_file_name)

    df['filename'] = new_name

    df.drop('level_0', inplace=True, axis=1)
    df.drop('index', inplace=True, axis=1)

    df['len_slice'] = df['end'].sub(df['start'], axis = 0)
    print(df.head(10))

    return df



def set_target_of_df(df, target = 'wheeze/crackle'):
    '''
    Parameters
    ----------
    df : pandas dataframe to be edited.
    target : What is the desired target. Can be set to 'crackle', 'diagnosis', 'wheeze/crackle'
    The default is 'crackle'.

    Returns
    -------
    new dataframe with the specified target.

    '''
    print(target)
    if not(target == 'crackle' or target == 'wheeze/crackle'):
        df = df.reset_index()
        return df

    ab = []
    for idx, row in df.iterrows():
        if (target == 'crackle'):
            if row['crackles'] == 1:
                ab.append('crackle')
            else:
                ab.append('no-crackle')

        elif (target == 'wheeze/crackle'):
            if (row['crackles'] == 1 and row['wheezes'] == 1):
                ab.append('both')
                continue
            elif row['crackles'] == 1:
                ab.append('crackle')
                continue
            elif (row['wheezes'] == 1):
                ab.append('wheeze')
                continue
            else:
                ab.append('none')
                continue

    df['abnormality'] = ab
    df.drop('wheezes', inplace=True, axis=1)
    df.drop('crackles', inplace=True, axis=1)
    df = df.reset_index()
    return df

def main():
    return 0

main()
