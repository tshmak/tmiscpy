import os
from os.path import join
import tempfile

import pandas as pd

from tmiscpy.audio_tools import WavAudio

def pipe2file(cmd, **kwargs):
    """
    Turn a bash pipe into a file like object that can be read by pd.read_csv()
    """
    import subprocess
    a = subprocess.Popen(cmd, stdout=subprocess.PIPE, **kwargs)
    from io import StringIO
    b = StringIO(a.communicate()[0].decode('utf-8'))
    return b

class KaldiAudio: 
    """
    A class for handling kaldi audio data 
    See https://kaldi-asr.org/doc/data_prep.html for background

    It outputs a pandas Series, or a pandas DataFrame if the .df() function is used 
    The DataFrame/Series is indexed by the utterance ID, and can be subsetted using standard 
    pandas syntax. 

    The utterance ID (uttID) is taken from the segments file. If the segments file is not present, 
    it comes from the wav.scp file (which must be present), and is equal to the recording ID (recID). 

    Usage examples: 
    s = KaldiAudio('/path/to/kaldi_data')
    s.uttID                                # Outputs a pandas Series
    s.recID
    s.begin                                # Note that this may not be available.
    s.wavfile 
    s.df(['uttID', 'recID', 'wavfile'])    # Outputs a pandas DataFrame
    s.wavfile.loc[some_uttID_list]         # for subsetting
    s.wavfile.reindex[some_uttID_list]         # for subsetting (but insert nan if uttID not in uttID)
    s.df(['wavfile', 'begin', 'end']).to_csv(header=False, index=False, sep='\t')
    """

    def __init__(self, segment_path, 
                 segments = 'segments', 
                 text = 'text', 
                 wav_scp = 'wav.scp', 
                 reco2file_and_channel = 'reco2file_and_channel', 
                 utt2spk = 'utt2spk', 
                 spk2utt = 'spk2utt', 
                 feats_scp = 'feats.scp', 
                 cmvn_scp = 'cmvn.scp'):
        
        print('KaldiAudio is deprecated. Use KaldiAudio2 if possible.')
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
                return pd.read_csv(pipefile, sep='\t', 
                        header=None, na_filter=False, **kwargs)
            else: 
                return pd.read_csv(txtfile, sep='\s+',
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
        return self._text.reindex(self.uttID)['text']
        
    @property
    def wavfile(self): 
        res = self._wav_scp.reindex(self.recID)['wavfile']
        res.index = self.uttID
        return res
        
    @property
    def begin(self):
        return self._segments.reindex(self.uttID)['begin']
        
    @property
    def end(self):
        return self._segments.reindex(self.uttID)['end']
    
    @property
    def sphfile(self):
        res = self._reco2file_and_channel.reindex(self.recID)['sphfile']
        res.index = self.uttID
        return res
    
    @property
    def rec_side(self):
        res = self._reco2file_and_channel.reindex(self.recID)['rec_side']
        res.index = self.uttID
        return res
    
    @property
    def spkID(self):
        return self._utt2spk.reindex(self.uttID)['spkID']
    
    @property
    def featsfile(self):
        return self._feats_scp.reindex(self.uttID)['featsfile']
    
    @property
    def cmvn_feats(self):
        res = self._cmvn_scp.reindex(self.spkID)['cmvn_feats']
        res.index = self.uttID
        return res
    
    def df(self, list_of_vars): 
        
        l = []
        for i in list_of_vars: 
            l = l + [self.__getattribute__(i)]
        return pd.concat(l, axis=1, join='inner')


class KaldiAudio2: 
    """
    A new class for handling kaldi audio data 
    Like KaldiAudio, I'll try to read a Kaldi directory
    into a pandas data_frame, with each row corresponding
    to a utterance
    However, unlike KaldiAudio, I'll output lists and 
    not pandas.Series
    """
    names = {
            "wav.scp": {
                "names": ["recID", "wavfile"], 
                "index_col": "recID",
                "two_cols": True,
                },
            "text": {
                "names": ["uttID", "text"],
                "index_col": "uttID",
                "two_cols": True,
                }, 
            "reco2file_and_channel": {
                "names": ["recID", "sphfile", "rec_side"],
                "index_col": "recID",
                }, 
            "utt2spk": {
                "names": ["uttID", "spkID"],
                "index_col": "uttID",
                }, 
            "spk2utt": {
                "names": ["spkID", "uttIDs"],
                "index_col": "spkID",
                "two_cols": True,
                }, 
            "spk2gender": {
                "names": ["spkID", "gender"],
                "index_col": "spkID",
                }, 
            "feats.scp": {
                "names": ["uttID", "featsfile"],
                "index_col": "uttID",
                "two_cols": True,
                }, 
            "cmvn.scp": {
                "names": ["spkID", "cmvn_feats"],
                "index_col": "spkID",
                "two_cols": True,
                }, 
            "segments": {
                "names": ["uttID", "recID", "begin", "end"],
                "index_col": "uttID",
                }, 
            }

    def __init__(self, basedir, output_pandas_series=False): 
        assert os.path.isfile(join(basedir, "wav.scp"))
        self.basedir = basedir
        self.columns = dict()
        self.data = dict()
        self.has_segments = os.path.isfile(join(basedir, 'segments')) 
        self.has_spks = os.path.isfile(join(basedir, 'utt2spk')) 
        self.index = dict()
        if self.has_segments:
            self.index['uttID'] = pd.Series(self.get_data('segments').index)
            self.index['recID'] = self.get_data('segments')['recID'] 
        else: 
            self.index['recID'] = pd.Series(self.get_data('wav.scp').index)
            self.index['uttID'] = pd.Series(self.index['recID'])
        if self.has_spks:
            self.index['spkID'] = self.get_data('utt2spk')['spkID']

        if output_pandas_series: 
            raise NotImplementedError # Not sure if this will be needed

        self.recordings = dict()

    
    def get_data(self, which):
        if which in self.data: 
            return self.data[which]
        self.data[which] = self.readtxt(
                join(self.basedir, which), 
                **self.names[which], 
                ) # Output: pandas.DataFrame
        return self.data[which]

    def readtxt(self, txtfile, two_cols=False, **kwargs): 

        if os.path.exists(txtfile): 
            if two_cols: 
                pipefile = pipe2file(['sed', 's/ /\t/', txtfile])
                return pd.read_csv(pipefile, sep='\t', 
                        header=None, na_filter=False, **kwargs)
            else: 
                return pd.read_csv(txtfile, sep='\s+',
                        header=None, na_filter=False, **kwargs)
        else: 
            raise ValueError(txtfile + ' does not exist.')

    def get_column(self, data, which, index='uttID'):
        self.check_dir_exists(data)
        assert self.get_data(data).index.name == index
        if which not in self.columns: 
            self.columns[which] = self.get_data(data).loc[self.index[index]][which].to_list()
        return self.columns[which]

    def check_dir_exists(self, which):
        if not os.path.isfile(join(self.basedir, which)):
            raise Exception(f'This kaldi directory does not have a {which} file: {self.basedir}')

    @property
    def begin(self): 
        return self.get_column('segments', 'begin')

    @property
    def end(self): 
        return self.get_column('segments', 'end')

    @property
    def wavfile(self): 
        return self.get_column('wav.scp', 'wavfile', index='recID')

    @property
    def recID(self):
        return self.index['recID']

    @property
    def uttID(self):
        return self.index['uttID']

    @property
    def text(self): 
        return self.get_column('text', 'text')

    @property
    def sphfile(self): 
        return self.get_column('reco2file_and_channel', 'sphfile', index='recID')

    @property
    def rec_side(self): 
        return self.get_column('reco2file_and_channel', 'rec_side', index='recID')

    @property
    def spkID(self):
        return self.get_column('utt2spk', 'spkID')

    @property
    def featsfile(self):
        return self.get_column('feats.scp', 'featsfile')
    
    @property
    def cmvn_feats(self):
        return self.get_column('cmvn.scp', 'cmvn_feats')

    @property
    def gender(self):
        return self.get_column('spk2gender', 'gender', index='spkID')

    def utterance(self, uttID): 
        if not self.has_segments:
            recID = uttID
        else:
            seg = self.get_data('segments').loc[uttID]
            recID = seg['recID']
        if recID not in self.recordings: 
            _wavfile = self.get_data('wav.scp')['wavfile'].loc[recID]
            if _wavfile.strip()[-1] == '|': # is a pipe rather than a .wav file
                wavfile = self.pipe_to_tempwav(_wavfile)
            elif _wavfile.strip().lower().endswith('.wav'):
                wavfile = _wavfile
            self.recordings[recID] = WavAudio(wavfile)
        if not self.has_segments: 
            return self.recordings[recID]
        else:
            utt = self.recordings[recID].segment(seg['begin'], seg['end'])
            return utt

    @staticmethod
    def pipe_to_tempwav(soxpipe): 
        raise NotImplementedError


if __name__ == '__main__': 
     path = '/mnt/nas2/asr_data/mandarin/boyue/original'
     s = KaldiAudio2(path)

