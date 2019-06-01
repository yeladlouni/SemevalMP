from abc import ABCMeta, abstractmethod
import os
import json

from torchtext.data.dataset import Dataset
from torchtext.data.example import Example

from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


class SemevalDataset(Dataset, metaclass=ABCMeta):

    # Child classes must define
    NAME = None
    NUM_CLASSES = None
    ID_FIELD = None
    TEXT_FIELD = None
    EXT_FEATS_FIELD = None
    LABEL_FIELD = None
    RAW_TEXT_FIELD = None
    EXT_FEATS = 4

    @abstractmethod
    def __init__(self, path):
        """
        Create a Castor dataset involving pairs of texts
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD),
                  ('ext_feats', self.EXT_FEATS_FIELD),
                  ('label', self.LABEL_FIELD), ('sentence_1_raw', self.RAW_TEXT_FIELD),
                  ('sentence_2_raw', self.RAW_TEXT_FIELD)]

        examples = []

        ids, labels, sent_list_1, sent_list_2 = [], [], [], []
        with open(path) as f:
            for line in f:
                content = json.loads(line)
                sent_list_1.append(content['question'])
                sent_list_2.append(content['qaquestion'])

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)
        self.word_to_doc_cnt = word_to_doc_cnt


        with open(path) as f:
            for line in f:
                content = json.loads(line)
                ids.append(content['qid'])
                labels.append(content['qarel'])

        for pair_id, l1, l2, ext_feats, label in zip(ids, sent_list_1, sent_list_2, overlap_feats, labels):
            example = Example.fromlist([pair_id, l1, l2, ext_feats, label, ' '.join(l1), ' '.join(l2)], fields)
            examples.append(example)

        super(SemevalDataset, self).__init__(examples, fields)
