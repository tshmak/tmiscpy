import os

import pandas as pd

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
        
        l = []
        for i in list_of_vars: 
            l = l + [self.__getattribute__(i)]
        return pd.concat(l, axis=1, join='inner')


 #path = '/home/tshmak/WORK/Projects/WFST/cmhkdemo1/data/cmhk_3rd_batch_test'
 #s = KaldiAudio(path)

