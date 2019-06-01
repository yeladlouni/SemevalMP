import torch
from torchtext.data.field import Field, RawField
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.castor_dataset import CastorPairDataset
from datasets.semeval_dataset import SemevalDataset


class CQA(SemevalDataset):
    NAME = 'wikiqa'
    NUM_CLASSES = 2
    ID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    RAW_TEXT_FIELD = RawField()

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path):
        """
        Create a WIKIQA dataset instance
        """
        super(CQA, self).__init__(path)

    @classmethod
    def splits(cls, path, train='train.json', validation='dev.json', test='test.json', **kwargs):
        return super(SemevalDataset, cls).splits(path, train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, validation, test = cls.splits(path)

        cls.TEXT_FIELD.build_vocab(train, validation, test, vectors=vectors)

        return BucketIterator.splits((train, validation, test), batch_size=batch_size, repeat=False, shuffle=shuffle, sort_within_batch=True, device=device)
