#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:08:02 2019

@author: tshmak
"""

#import os
import scipy.io.wavfile 
import matplotlib.pyplot as plt
import wave
import tempfile
import subprocess
import math
import numpy as np
#import pdb
#import time


class wavaudio: 
    def __init__(self, wavfile): 
        self.wavfile = wavfile
        opened_wav = wave.open(self.wavfile, 'rb')
        
        self.rate = opened_wav.getframerate()
        self.nframes = opened_wav.getnframes()
        self.duration = self.nframes / self.rate
        self.sampwidth = opened_wav.getsampwidth()
        
        self.params = opened_wav.getparams()
        opened_wav.close()
        
        
        
    def adjust_frames(self, startframe: int, endframe: int): 
        length = endframe - startframe
        remainder = length % self.sampwidth
        return endframe - remainder
    
    def _save_segment_to_file(self, startframe: int, endframe: int, file = tempfile.NamedTemporaryFile(suffix='.wav').name): 
        
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
        subprocess.run(['play', wavfile])
        
    def _play_segment(self, startframe: int, endframe: int): 
        wavfile = self._save_segment_to_file(startframe, endframe)
        self.play_wav(wavfile)
        
    def sec2frame(self, sec: float):
        frames = math.floor(sec * self.rate)
        return frames
    
    def play_segment(self, start_sec: float, end_sec: float): 
        self._play_segment(self.sec2frame(start_sec), self.sec2frame(end_sec))
        
    def save_segment_to_file(self, start_sec: float, end_sec: float, file = tempfile.NamedTemporaryFile(suffix='.wav').name):
        return self._save_segment_to_file(self.sec2frame(start_sec), self.sec2frame(end_sec), file)
        
    def wav2int(self, wavfile): 
        return scipy.io.wavfile.read(wavfile)
    
    def plotwav(self, wavfile, start = 0): 
        wa = wavaudio(wavfile)
        y = self.wav2int(wavfile)[1]
        x = np.linspace(start,start+wa.duration, wa.nframes)
        plt.clf()
        plt.plot(x, y)
        plt.xlabel('seconds')
        
    def plot_segment(self, start_sec: float, end_sec: float): 
        tempwav = self.save_segment_to_file(start_sec, end_sec)
        self.plotwav(tempwav, start = start_sec)
        
    def plot_n_play(self, start_sec: float, end_sec: float, plot_start = None, plot_end = None): 
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