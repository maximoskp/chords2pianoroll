import numpy as np
from transformers import PreTrainedTokenizerBase
# https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils

class BinaryTokenizer(PreTrainedTokenizerBase):
    def __init__(self, num_digits=12):
        self.vocab_size = 2**12
    # end init
    
    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus):
        return corpus.dot(1 << np.arange(corpus.shape[-1] - 1, -1, -1))
    # end transform

    def fit_transform(self, corpus):
        return corpus.dot(1 << np.arange(corpus.shape[-1] - 1, -1, -1))
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__
# end class BinaryTokenizer

class SimpleSerialChromaTokenizer(PreTrainedTokenizerBase):
    def __init__(self, max_num_segments=0, pad_to_length=0, model_max_length=0):
        '''
        0: padding
        1: beginning of sequence
        2: end of sequence
        3 to 14: pitch classes
        15 to (15+max_num_segments): new chroma segment with index
        '''
        self.pad_token = 'pad'
        self.bos_token = 'bos'
        self.eos_token = 'eos'
        self.chord_offset = 3
        self.max_num_segments = max_num_segments
        self.segment_offset = 15
        self.pad_to_length = pad_to_length
        self.vocab_size = self.segment_offset + max_num_segments
        self.model_max_length = model_max_length # to be updated as data are tokenized or set by hand for online data
        self.vocab = {
            'pad': 0,
            'bos': 1,
            'eos': 2
        }
        for i in range(3, 15, 1):
            self.vocab['c_' + str(i-3)] = i
        for i in range( 0, max_num_segments, 1 ):
            self.vocab['seg_' + str(i)] = i
    # end init
    
    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus):
        # check if multiple pianorolls are given
        multiple_pianorolls = False
        if type(corpus) is list or len(corpus.shape) == 3:
            multiple_pianorolls = True
        if multiple_pianorolls:
            serialized = []
            ids = []
            for i in corpus:
                serialized_tmp, ids_tmp = self.sequence_serialization(corpus[i])
                serialized.append( serialized_tmp )
                ids.append( ids_tmp )
        else:
            serialized, ids = self.sequence_serialization(corpus)
        return {'tokens': serialized, 'input_ids': ids}
    # end transform

    def fit_transform(self, corpus):
        return self.transform(corpus)
    # end transform

    def __call__(self, corpus):
        return self.transform(corpus)
    # end __call__

    def sequence_serialization(self, chroma):
        tokens = []
        ids = []
        segment_idx = 0
        tokens.append( self.bos_token)
        ids.append( self.vocab[ self.bos_token ] )
        tokens.append( 'seg_' + str( segment_idx ) )
        ids.append(self.segment_offset + segment_idx)
        if self.max_num_segments > 0:
            segment_idx += 1
            segment_idx = segment_idx % self.max_num_segments
        for i in range(chroma.shape[0]):
            if self.max_num_segments > 0:
                segment_idx += 1
                segment_idx = segment_idx % self.max_num_segments
            # check if chord pcs exist
            c = chroma[i,:]
            nzc = np.nonzero(c)[0]
            for i in range(nzc.shape[0]):
                tokens.append( 'c_' + str(nzc[i]) )
                ids.append( int(nzc[i] + self.chord_offset) )
            tokens.append( 'seg_' + str( segment_idx ) )
            ids.append(self.segment_offset + segment_idx)
        return tokens, ids
    # end sequence_serialization
# end class SimpleSerialChromaTokenizer

class SimpleChromaSerializer:
    def __init__(self, max_num_segments=0, pad_to_length=0):
        '''
        0: padding
        1: start of sequence
        2: end of sequence
        3 to 14: pitch classes
        15 to (15+max_num_segments): new chroma segment with index
        '''
        self.padding = 0
        self.start_of_sequence = 1
        self.end_of_sequence = 2
        self.max_num_segments = max_num_segments
        self.segment_offset = 15
        self.pad_to_length = pad_to_length
    # end init

    def sequence_serialization(self, chroma):
        seq_in = []
        segment_idx = 0
        seq_in.append(self.segment_offset + segment_idx)
        seq_in.append(self.segment_offset + segment_idx)
        if self.max_num_segments > 0:
            segment_idx += 1
            segment_idx = segment_idx % self.max_num_segments
        for i in range(chroma.shape[0]):
            # check if chord pcs exist
            c = chroma[i,:]
            # check if no more chords
            if np.sum( chroma[i:,:] ) > 0:
                nzc = np.nonzero(c)[0]
                seq_in.extend( nzc + self.chord_offset )
            seq_in.append(self.segment_offset + segment_idx)
            if self.max_num_segments > 0:
                segment_idx += 1
                segment_idx = segment_idx % self.max_num_segments
        return seq_in
    # end sequence_serialization
    
# end class