#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:08:02 2019

@author: tshmak
"""

__all__ = ['wavaudio', 'read_mnist', 'examine', 'jiebacut', 'chdir', 
           'home', 'os', 'head', 'interactive']
import pdb
import os

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
    if sys.argv[0] == '': 
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
        
        
    
#os.chdir('/Users/tshmak/WORK/Projects/segmentation/testaudio') 
#wavfile = '181221_1253_mono.wav'
#test = wavaudio(wavfile)        
#test.plot_n_play(0,2, 0,10)
#test.plot_n_play(2,4, 0,10)