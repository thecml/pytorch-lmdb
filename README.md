# pytorch-lmdb
Forked from https://github.com/Lyken17/Efficient-PyTorch/ and simplified. Also works on Windows systems now.

# Speed overview
Trained on the Cats versus Dogs dataset avaliable on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). Results compare the torch.ImageFolder and our lmdb implementation.

```
Timings for lmdb (min egen implementering)
Avg data time: 0.011866736168764075
Avg batch time: 0.10090051865091129
Total data time: 2.325880289077759
Total batch time: 19.776501655578613

Timings for imagefolder: 
Avg data time: 0.017892257291443493 
Avg batch time: 0.1053010200967594  
Total data time: 3.506882429122925  
Total batch time: 20.638999938964844
```

# LMDB 
The format of converted LMDB is defined as follow.

key | value 
--- | ---
img-id1 | (jpeg_raw1, label1)
img-id2 | (jpeg_raw2, label2)
img-id3 | (jpeg_raw3, label3)
... | ...
img-idn | (jpeg_rawn, labeln)
`__keys__` | [img-id1, img-id2, ... img-idn]
`__len__` | n

As for details of reading/writing, please refer to [code](folder2lmdb.py).


## Convert `ImageFolder` to `LMDB`
```bash
python folder2lmdb.py -f ~/torch_data/ -s train
```

## ImageFolderLMDB
The usage of `ImageFolderLMDB` is identical to `torchvision.datasets`. 

```python
import ImageFolderLMDB
from torch.utils.data import DataLoader
dst = ImageFolderLMDB(path, transform, target_transform)
loader = DataLoader(dst, batch_size=64)
```

