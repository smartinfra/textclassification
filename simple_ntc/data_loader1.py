from torchtext.legacy import data
from konlpy.tag import Mecab


class DataLoader(object):
    '''
    Data loader class to load text file using torchtext library.
    '''

    def __init__(
        self, train_fn,
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        use_eos=False,
        shuffle=True,
    ):
        '''
        DataLoader initialization.
        :param train_fn: Train-set filename
        :param batch_size: Batchify data fot certain batch size.
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: Maximum vocabulary size
        :param min_freq: Minimum frequency for loaded word.
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super().__init__()
        # Mecab을 토크나이저로 사용
        tokenizer = Mecab()
        # Define field of the input file.
        # The input file consists of two fields.
        self.text = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs, # 토크나이저로는 Mecab 사용.
                  lower=True,
                  batch_first=True,
                  fix_length=20)

        self.label = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True)
        
        # Those defined two columns will be delimited by TAB.
        # Thus, we use TabularDataset to load two columns in the input file.
        # We would have two separate input file: train_fn, valid_fn
        # Files consist of two columns: label field and text field.
        
        
        train, valid = data.TabularDataset(
            path=train_fn,
            format='csv', 
            fields=[
                ('comments', self.text),
                ('label', self.label),    
            ],
            skip_header=True
        ).split(split_ratio=(1 - valid_ratio))
        print('훈련 샘플의 개수 : {}'.format(len(train)))
        print(vars(train[0]))
        self.label.build_vocab(train)
        print(len(self.label.vocab))
        # Those loaded dataset would be feeded into each iterator:
        # train iterator and valid iterator.
        # We sort input sentences by length, to group similar lengths.
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            repeat=False
        )

        # At last, we make a vocabulary for label and text field.
        # It is making mapping table between words and indice.
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)
        print(len(self.label.vocab))
        print(len(self.text.vocab))
