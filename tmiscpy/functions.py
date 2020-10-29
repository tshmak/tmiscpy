#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:08:02 2019

@author: tshmak
"""

__all__ = ['wavaudio', 'read_mnist', 'examine', 'jiebacut', 'chdir', 
           'home', 'os', 'head', 'interactive', 'filename', 'kaldi_audio']
import pdb
import os, sys


"""
like my old Rfilename command in R
"""
def filename(filenamestub): 
    import sys
    import re
    filename = sys.argv[0]
    if (re.search('python$', filename) != None) | (filename == ''): 
        filename = filenamestub
        
    filenamestub = re.sub('\\.py$', '', filename)

    return filenamestub


"""
Turn a bash pipe into a file like object that can be read by pd.read_csv()
"""
def pipe2file(cmd, **kwargs):
    
    import subprocess
    
    a = subprocess.Popen(cmd, stdout=subprocess.PIPE, **kwargs)
    
    from io import StringIO
    
    b = StringIO(a.communicate()[0].decode('utf-8'))
    
    return b



"""
Test if interactive
"""
def interactive(): 
    import sys
    return hasattr(sys, 'ps1')


"""
To get a view of an object 
"""
def head(obj, n=10, tail=False, sample=False): 
    if type(obj) is list: 
        if not sample: 
            if tail:
                print('Not yet implemented')
            else: 
                print('Not yet implemented')
        else: 
            print('Not yet implemented')
    else: 
        print('Not yet implemented')
        


"""
Get home directory
"""
def home(): 
    from os.path import expanduser
    home = expanduser("~")
    return home

"""
Change current directory (if in interactive mode)
"""
def chdir(dir: str):
    import sys, os
    if interactive(): 
        os.chdir(dir)

"""
Cut strings using jieba 
https://github.com/fxsjy/jieba
**** Actually this function is redundant... just use jieba.lcut() **** 
**** Actually it handles in a different way from jieba.lcut() ****
"""
def jiebacut(string: str): 
    import jieba
    if type(string) != str: 
        print('string needs to be of type str. Try using ' + 
              '\'",".join(str(x) for x in string) \'' + 
              'to concatenate them.')
        return None
    
    jb = jieba.cut(string)
    words = []
    while True: 
        try: 
            test = next(jb)
            if test != ',': 
                words.append(test)
        except: 
            break     
    return words

""" examine
Trying to replicate R's str() function
""" 

def examine(obj, TYPE = '', PRINT=True): 
    from tabulate import tabulate
        
    Dict = {}
    A = dir(obj)
    for item in A:
#        pdb.set_trace()
        cmd = 'type(obj.' + item + ').__name__'
        Type = eval(cmd)
        
        if TYPE != '': 
            if Type != TYPE: 
                continue
            
        Dict[item] = Type
        
    if (PRINT): 
        res = tabulate(Dict.items(), headers=['Object', 'Type'], tablefmt='orgtbl')
        print(res)
    else: 
        return Dict


""" read_mnist
Script for reading MNIST dataset.
Script downloaded from https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
!! Remember to ungzip the file first! 
"""


def read_mnist(filename):
    import struct
    import numpy as np
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


""" wavaudio
A class for dealing with .wav files, including functions for playing and plotting
"""

class wavaudio: 
    #import os
    import tempfile
    #import time

    def __init__(self, wavfile): 
        import wave
        self.wavfile = wavfile
        opened_wav = wave.open(self.wavfile, 'rb')
        
        self.rate = opened_wav.getframerate()
        self.nframes = opened_wav.getnframes()
        self.duration = self.nframes / self.rate
        self.sampwidth = opened_wav.getsampwidth()
        
        self.params = opened_wav.getparams()
        self.play_options = None
        opened_wav.close()
        
        
        
    def adjust_frames(self, startframe: int, endframe: int): 
        length = endframe - startframe
        remainder = length % self.sampwidth
        return endframe - remainder
    
    def _save_segment_to_file(self, startframe: int, endframe: int, file = tempfile.NamedTemporaryFile(suffix='.wav').name): 
        import wave
        
        endframe = self.adjust_frames(startframe, endframe) # Remove the incomplete samples
        
        opened_wav = wave.open(self.wavfile, 'rb')
        opened_wav.setpos(startframe)
        blob = opened_wav.readframes(endframe - startframe)
        opened_wav.close()
        
        out = wave.open(file, 'wb')
        out.setnchannels(self.params.nchannels)
        out.setsampwidth(self.params.sampwidth)
        out.setframerate(self.params.framerate)
        out.writeframes(blob)
        out.close()
        
        return file
    
    def play_wav(self, wavfile): 
        # Requires 'sox' installed externally 
        import subprocess
        if self.play_options is not None: 
            subprocess.run(['play', wavfile, self.play_options])
        else:
            subprocess.run(['play', wavfile])
        
    def _play_segment(self, startframe: int, endframe: int): 
        wavfile = self._save_segment_to_file(startframe, endframe)
        self.play_wav(wavfile)
        
    def sec2frame(self, sec: float):
        import math
        frames = math.floor(sec * self.rate)
        return frames
    
    def play_segment(self, start_sec: float, end_sec: float): 
        self._play_segment(self.sec2frame(start_sec), self.sec2frame(end_sec))
        
    def save_segment_to_file(self, start_sec: float, end_sec: float, file = tempfile.NamedTemporaryFile(suffix='.wav').name):
        return self._save_segment_to_file(self.sec2frame(start_sec), self.sec2frame(end_sec), file)
        
    def wav2int(self, wavfile): 
        import scipy.io.wavfile 
        return scipy.io.wavfile.read(wavfile)
    
    def plotwav(self, wavfile, start = 0): 

        import matplotlib.pyplot as plt
        import numpy as np

        wa = wavaudio(wavfile)
        y = self.wav2int(wavfile)[1]
        x = np.linspace(start,start+wa.duration, wa.nframes)
        plt.clf()
        plt.plot(x, y, linewidth=0.1)
        plt.xlabel('seconds')
        
    def plot_segment(self, start_sec: float, end_sec: float): 
        tempwav = self.save_segment_to_file(start_sec, end_sec)
        self.plotwav(tempwav, start = start_sec)
        
    def plot_n_play(self, start_sec: float, end_sec: float, plot_start = None, plot_end = None): 

        import matplotlib.pyplot as plt

        length = end_sec - start_sec
        if plot_start is None: 
            plot_start = max(0, start_sec - length)
            
        if plot_end is None: 
            plot_end = min(self.duration, end_sec + length)
        
        self.plotwav(self.save_segment_to_file(plot_start, plot_end), start=plot_start)
        plt.axvspan(start_sec, end_sec, color='gray', alpha=0.5)
        plt.pause(0.1) # For some reason this is needed to force python the render the graph immediately rather than when the function ends
        # https://stackoverflow.com/questions/37999928/how-to-force-matplotlib-to-plot-before-the-end-of-a-function

        self.play_segment(start_sec, end_sec)
        

"""
A class for handling kaldi audio data 
See https://kaldi-asr.org/doc/data_prep.html for background

It outputs a pandas Series, or a pandas DataFrame if the .df() function is used 
The DataFrame/Series is indexed by the utterance ID, and can be subsetted using standard 
pandas syntax. 

The utterance ID (uttID) is taken from the segments file. If the segments file is not present, 
it comes from the wav.scp file (which must be present), and is equal to the recording ID (recID). 

Usage examples: 
s = kaldi_audio('/path/to/kaldi_data')
s.uttID                                # Outputs a pandas Series
s.recID
s.begin                                # Note that this may not be available.
s.wavfile 
s.df(['uttID', 'recID', 'wavfile'])    # Outputs a pandas DataFrame
s.wavfile.loc[some_uttID_list]         # for subsetting
s.df(['wavfile', 'begin', 'end']).to_csv(header=False, index=False, sep='\t')
"""

import pandas as pd
class kaldi_audio: 
    
    def __init__(self, segment_path, 
                 subset = None, 
                 segments = 'segments', 
                 text = 'text', 
                 wav_scp = 'wav.scp', 
                 reco2file_and_channel = 'reco2file_and_channel', 
                 utt2spk = 'utt2spk', 
                 spk2utt = 'spk2utt', 
                 feats_scp = 'feats.scp', 
                 cmvn_scp = 'cmvn.scp'):
        
        self.segments_file = segment_path + '/' + segments
        self.text_file = segment_path + '/' + text
        self.reco2file_and_channel_file = segment_path + '/' + reco2file_and_channel
        self.wav_scp_file = segment_path + '/' + wav_scp
        self.utt2spk_file = segment_path + '/' + utt2spk
        self.spk2utt_file = segment_path + '/' + spk2utt
        self.feats_scp_file = segment_path + '/' + feats_scp
        self.cmvn_scp_file = segment_path + '/' + cmvn_scp

        # wav_scp MUST be present. Define recID
        self._wav_scp = self.readtxt(self.wav_scp_file, two_cols=True, names=['recID', 'wavfile'], index_col='recID')

        # segments file is OPTIONAL. Define uttID
        try: 
            self._segments = self.readtxt(self.segments_file, names=['uttID', 'recID', 'begin', 'end'], index_col='uttID')
            self.uttID = self._segments.index.to_series()
            self.recID = self._segments['recID']
        except ValueError: 
            self.recID = self._wav_scp.index.to_series()
            self.recID.name = 'uttID'
            self.uttID = self.recID

        #self.uttID

    # The all-important readtxt function 
    def readtxt(self, txtfile, two_cols=False, **kwargs): 
        if os.path.exists(txtfile): 
            if two_cols: 
                pipefile = pipe2file(['sed', 's/ /\t/', txtfile])
                return pd.read_csv(pipefile, sep='\t', header=None, **kwargs)
            else: 
                return pd.read_csv(txtfile, delim_whitespace=True, header=None, **kwargs)
        else: 
            raise ValueError(txtfile + ' does not exist.')

    # Various files read as pd.DataFrame
    @property
    def _text(self): 
        return self.readtxt(self.text_file, two_cols=True, names=['uttID', 'text'], index_col='uttID')

    @property
    def _reco2file_and_channel(self): 
        return self.readtxt(self.reco2file_and_channel_file, names=['recID', 'sphfile', 'rec_side'], index_col='recID')

    @property
    def _utt2spk(self):
        return self.readtxt(self.utt2spk_file, names=['uttID', 'spkID'], index_col='uttID')

    @property
    def _spk2utt(self): 
        return self.readtxt(self.spk2utt_file, two_cols=True, names=['spkID', 'uttIDs'], index_col='spkID')

    @property
    def _feats_scp(self):
        return self.readtxt(self.feats_scp_file, two_cols=True, names=['uttID', 'featsfile'], index_col='uttID')
    
    @property
    def _cmvn_scp(self):
        return self.readtxt(self.cmvn_scp_file, two_cols=True, names=['spkID', 'cmvn_feats'], index_col='spkID')
    
    # Various columns from the files as pd.Series (all indexed by uttID)
    @property
    def text(self): 
        return self._text.loc[self.uttID, 'text']
        
    @property
    def wavfile(self): 
        res = self._wav_scp.loc[self.recID, 'wavfile']
        res.index = self.uttID
        return res
        
    @property
    def begin(self):
        return self._segments.loc[self.uttID, 'begin']
        
    @property
    def end(self):
        return self._segments.loc[self.uttID, 'end']
    
    @property
    def sphfile(self):
        res = self._reco2file_and_channel.loc[self.recID, 'sphfile']
        res.index = self.uttID
        return res
    
    @property
    def rec_side(self):
        res = self._reco2file_and_channel.loc[self.recID, 'rec_side']
        res.index = self.uttID
        return res
    
    @property
    def spkID(self):
        return self._utt2spk.loc[self.uttID, 'spkID']
    
    @property
    def featsfile(self):
        return self._feats_scp.loc[self.uttID, 'featsfile']
    
    @property
    def cmvn_feats(self):
        res = self._cmvn_scp.loc[self.spkID, 'cmvn_feats']
        res.index = self.uttID
        return res
    
    def df(self, list_of_vars): 
        l = []
        for i in list_of_vars: 
            l = l + [self.__getattribute__(i)]
        return pd.concat(l, axis=1, join='inner')


 #path = '/home/tshmak/WORK/Projects/WFST/cmhkdemo1/data/cmhk_3rd_batch_test'
 #s = kaldi_audio(path)
