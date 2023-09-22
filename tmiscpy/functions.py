#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:08:02 2019

@author: tshmak
"""

__all__ = ['wavaudio', 'read_mnist', 'examine', 'jiebacut', 'chdir', 
           'home', 'os', 'head', 'interactive', 'filename', 'kaldi_audio', 
           'wavaudio2',
           ]
import pdb
import os, sys


def filename(filenamestub): 
    """
    like my old Rfilename command in R
    """
    import re
    filename = sys.argv[0]
    if (re.search('python$', filename) != None) | (filename == ''): 
        filename = filenamestub
        
    filenamestub = re.sub('\\.py$', '', filename)

    return filenamestub


def pipe2file(cmd, **kwargs):
    """
    Turn a bash pipe into a file like object that can be read by pd.read_csv()
    """
    
    import subprocess
    
    a = subprocess.Popen(cmd, stdout=subprocess.PIPE, **kwargs)
    
    from io import StringIO
    
    b = StringIO(a.communicate()[0].decode('utf-8'))
    
    return b



def interactive(): 
    """
    Test if interactive
    """
    import sys
    return hasattr(sys, 'ps1')


def head(obj, n=10, tail=False, sample=False): 
    """
    To get a view of an object 
    """
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
        


def home(): 
    """
    Get home directory
    """
    from os.path import expanduser
    home = expanduser("~")
    return home

def chdir(dir: str):
    """
    Change current directory (if in interactive mode)
    """
    if interactive(): 
        os.chdir(dir)

def jiebacut(string: str): 
    """
    Cut strings using jieba 
    https://github.com/fxsjy/jieba
    **** Actually this function is redundant... just use jieba.lcut() **** 
    **** Actually it handles in a different way from jieba.lcut() ****
    """
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

def examine(obj, TYPE = '', PRINT=True): 
    """ examine
    Trying to replicate R's str() function
    """ 
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

def read_mnist(filename):
    """ read_mnist
    Script for reading MNIST dataset.
    Script downloaded from https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    !! Remember to ungzip the file first! 
    """
    import struct
    import numpy as np
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


class wavaudio: 
    """ wavaudio
    A class for dealing with .wav files, including functions for playing and plotting
    """
    import tempfile

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
        self.playcmd = 'play'
        opened_wav.close()
        
    def adjust_frames(self, startframe: int, endframe: int): 
        length = endframe - startframe
        remainder = length % self.sampwidth
        return endframe - remainder
    
    def _save_segment_to_file(self, startframe: int, endframe: int, file = tempfile.NamedTemporaryFile(suffix='.wav').name): 
        import wave
        
        endframe = self.adjust_frames(startframe, endframe) # Remove the incomplete samples
        
        with wave.open(self.wavfile, 'rb') as opened_wav: 
            opened_wav.setpos(startframe)
            blob = opened_wav.readframes(endframe - startframe)
        
        with wave.open(file, 'wb') as out: 
            out.setnchannels(self.params.nchannels)
            out.setsampwidth(self.params.sampwidth)
            out.setframerate(self.params.framerate)
            out.writeframes(blob)
        
        return file
    
    def play_wav(self, wavfile): 
        # Requires 'sox' installed externally 
        import subprocess
        if self.play_options is not None: 
            subprocess.run([self.playcmd, wavfile, self.play_options])
        else:
            subprocess.run([self.playcmd, wavfile])
        
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
        
import tempfile
import wave
import os
import math
import re
import time
class wavaudio2: 
    """ wavaudio2
    A improved class over wavaudio
    """

    def __init__(self, wavfile, playcmd='play @wavfile', 
            sleep_after_play=False, _empty=False): 

        if _empty: # Not supposed to be called directly
            return

        from scipy.io import wavfile as spwav

        #self.wavfile = wavfile
        samp_rate, array = spwav.read(wavfile)
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

        wa = wavaudio2(None, _empty=True)
        wa._fill(array, samp_rate, playcmd, sleep_after_play)

        return wa

    def copy(self): 
        return wavaudio2.from_numpy(self.array.copy(), self.samp_rate, 
                               self.playcmd, self.sleep_after_play) 
        
    def write(self, wavfile=tempfile.NamedTemporaryFile(suffix='.wav').name): 
        
        from scipy.io import wavfile as spwav

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
        wa = wavaudio2.from_numpy(array, self.samp_rate, self.playcmd, 
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

        import matplotlib.pyplot as plt
        import numpy as np

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

        import matplotlib.pyplot as plt

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

        import numpy as np

        if self.dtype == np.int16: 
            return (self.array / 32768).astype(np.float32)
        elif isinstance(self.dtype, np.floating): 
            return self.array.astype(np.float32)
        else: 
            raise NotImplementedError

    def melspec(self, **kwargs): 

        import librosa

        if self.nchannels > 1: 
            raise NotImplementedError
        
        y = self.array_float32()[0]
        S = librosa.feature.melspectrogram(y=y, sr=self.samp_rate, **kwargs)
        return S

    def _plot_melspec(self, startframe, endframe, start=0, **kwargs): 

        import librosa
        import librosa.display
        import numpy as np
        import matplotlib.pyplot as plt

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

        import librosa

        if self.nchannels > 1: 
            raise NotImplementedError
        
        y = self.array_float32()[0]
        S = librosa.stft(y=y, **kwargs)
        return S

    def _plot_spec(self, startframe, endframe, start=0, **kwargs): 

        import librosa
        import librosa.display
        import numpy as np
        import matplotlib.pyplot as plt

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


class kaldi_audio: 
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

        import pandas as pd

        if os.path.exists(txtfile): 
            if two_cols: 
                pipefile = pipe2file(['sed', 's/ /\t/', txtfile])
                return pd.read_csv(pipefile, sep='\t', 
                        header=None, na_filter=False, **kwargs)
            else: 
                return pd.read_csv(txtfile, delim_whitespace=True,
                        header=None, na_filter=False, **kwargs)
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
        
        import pandas as pd

        l = []
        for i in list_of_vars: 
            l = l + [self.__getattribute__(i)]
        return pd.concat(l, axis=1, join='inner')


 #path = '/home/tshmak/WORK/Projects/WFST/cmhkdemo1/data/cmhk_3rd_batch_test'
 #s = kaldi_audio(path)
