#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tshmak
"""

import os, sys
import tempfile
import math
import re
import time

from scipy.io import wavfile as spwav
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class WavAudio: 
    """ WavAudio
    A improved class over wavaudio
    """

    def __init__(self, wavfile, samp_rate=None, playcmd='play @wavfile', 
            sleep_after_play=False, _empty=False): 

        if _empty: # Not supposed to be called directly
            return

        #self.wavfile = wavfile
        array, samp_rate = librosa.load(wavfile, sr=samp_rate)
        self._fill(array, samp_rate, playcmd, sleep_after_play)

    def _fill(self, array, samp_rate, playcmd, sleep_after_play): 
        if len(array.shape) == 1: 
            array = array[None,:]
        assert len(array.shape) == 2
        self.array = array
        self.samp_rate = samp_rate
        self.playcmd = playcmd
        self.sleep_after_play = sleep_after_play

    @property
    def nchannels(self):
        return self.array.shape[0]

    @property
    def nframes(self):
        return self.array.shape[1]

    @property
    def duration(self): 
        return self.nframes / self.samp_rate

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def itemsize(self):
        return self.array.itemsize

    def from_numpy(array, samp_rate, playcmd='play @wavfile', 
            sleep_after_play=False,
            file=tempfile.NamedTemporaryFile(suffix='.wav').name): 

        wa = WavAudio(None, _empty=True)
        wa._fill(array, samp_rate, playcmd, sleep_after_play)

        return wa

    def copy(self): 
        return WavAudio.from_numpy(self.array.copy(), self.samp_rate, 
                               self.playcmd, self.sleep_after_play) 
        
    def write(self, wavfile=tempfile.NamedTemporaryFile(suffix='.wav').name): 
        
        spwav.write(wavfile, self.samp_rate, self.array.T)
        return wavfile

    def _get_default_endframe(self, endframe):
        if endframe is None:
            return self.nframes
        else: 
            return min(self.nframes, endframe)

    def _get_play_cmd(self, wavfile: str):
        cmd = re.sub('@wavfile', wavfile, self.playcmd)
        if cmd == self.playcmd: 
            cmd = self.playcmd + ' ' + wavfile
        return cmd

    def _segment(self, startframe, endframe): 
        array = self.array[:, startframe:endframe]
        wa = WavAudio.from_numpy(array, self.samp_rate, self.playcmd, 
                self.sleep_after_play)
        return wa

    def segment(self, start_sec: float = 0.0, end_sec: float = None): 
        startframe = self.sec2frame(start_sec)
        endframe = self.sec2frame(end_sec)
        endframe = self._get_default_endframe(endframe)
        return self._segment(startframe, endframe)

    def play_wav(self, wavfile): 
        cmd = self._get_play_cmd(wavfile)
        os.system(cmd)
        
    def _play(self): 
        wavfile = self.write()
        self.play_wav(wavfile)
        if self.sleep_after_play:
            time.sleep(self.duration)

    def play(self, start_sec: float = 0.0, end_sec: float = None): 
        ss = self.segment(start_sec, end_sec)
        ss._play()

    def sec2frame(self, sec: float):
        if sec is not None:
            return math.floor(sec * self.samp_rate)
        else: 
            return None
    
    def _plot_wav(self, startframe, endframe, start = 0): 

        nframes = endframe - startframe
        duration = nframes / self.samp_rate
        if self.nchannels == 1: 
            x = np.linspace(start,start+duration, nframes)
            y = self.array[0, startframe:endframe]
            #plt.clf()
            plt.plot(x, y, linewidth=0.1)
            plt.xlabel('seconds')
        else: 
            raise NotImplementedError('_plot_wav() not yet implemented for > 1 channels')

    def plot(self, start_sec: float = 0.0, end_sec: float = None): 
        startframe = self.sec2frame(start_sec)
        endframe = self.sec2frame(end_sec)
        endframe = self._get_default_endframe(endframe)
        self._plot_wav(startframe, endframe, start = start_sec)
        
    def plot_n_play(self, start_sec: float = 0.0, end_sec: float = None, 
            plot_start = None, plot_end = None): 

        if end_sec is None: 
            _end_sec = self.duration
        else:
            _end_sec = end_sec
        length = _end_sec - start_sec

        if plot_start is None: 
            plot_start = max(0, start_sec - length)
            
        if plot_end is None: 
            plot_end = min(self.duration, _end_sec + length)
        
        self.plot(plot_start, plot_end)
        plt.axvspan(start_sec, _end_sec, color='gray', alpha=0.5)
        plt.pause(0.1) # For some reason this is needed to force python the render the graph immediately rather than when the function ends
        # https://stackoverflow.com/questions/37999928/how-to-force-matplotlib-to-plot-before-the-end-of-a-function

        self.play(start_sec, end_sec)

    def array_float32(self): 

        if self.dtype == np.int16: 
            return (self.array / 32768).astype(np.float32)
        elif isinstance(self.dtype, np.floating): 
            return self.array.astype(np.float32)
        else: 
            raise NotImplementedError

    def melspec(self, **kwargs): 

        if self.nchannels > 1: 
            raise NotImplementedError
        
        y = self.array_float32()[0]
        S = librosa.feature.melspectrogram(y=y, sr=self.samp_rate, **kwargs)
        return S

    def _plot_melspec(self, startframe, endframe, start=0, **kwargs): 

        nframes = endframe - startframe
        duration = nframes / self.samp_rate
        S = self._segment(startframe, endframe).melspec(**kwargs)
        SS = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(SS, y_axis='mel', fmax=8000, x_axis='time',
                x_coords=np.linspace(start, start+duration, SS.shape[1]))
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()

    def plot_melspec(self, start_sec: float = 0.0, end_sec: float = None, **kwargs): 
        startframe = self.sec2frame(start_sec)
        endframe = self.sec2frame(end_sec)
        endframe = self._get_default_endframe(endframe)
        self._plot_melspec(startframe, endframe, start = start_sec, **kwargs)

    def stft(self, **kwargs): 

        if self.nchannels > 1: 
            raise NotImplementedError
        
        y = self.array_float32()[0]
        S = librosa.stft(y=y, **kwargs)
        return S

    def _plot_spec(self, startframe, endframe, start=0, **kwargs): 

        nframes = endframe - startframe
        duration = nframes / self.samp_rate
        S = self._segment(startframe, endframe).stft(**kwargs)
        SS = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(SS, y_axis='log', x_axis='time', 
                x_coords=np.linspace(start, start+duration, SS.shape[1]))
        plt.colorbar(format='%+2.0f dB')
        plt.title('Power Spectrogram')
        plt.tight_layout()
        #plt.xlim([start, start + duration])

    def plot_spec(self, start_sec: float = 0.0, end_sec: float = None, **kwargs): 
        startframe = self.sec2frame(start_sec)
        endframe = self.sec2frame(end_sec)
        endframe = self._get_default_endframe(endframe)
        self._plot_spec(startframe, endframe, start = start_sec, **kwargs)


