import torch.utils.data as data
import io
import pickle
import lmdb
import os.path as osp
from PIL import Image

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform
        
        env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        
    def open_lmdb(self):
         self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                              readonly=True, lock=False,
                              readahead=False, meminit=False)
         self.txn = self.env.begin(write=False, buffers=True)
         self.length = pickle.loads(self.txn.get(b'__len__'))
         self.keys = pickle.loads(self.txn.get(b'__keys__'))

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        
        img, target = None, None
        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = io.BytesIO()
        buf.write(imgbuf[0])
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'