# pytorch-lmdb
Forked from https://github.com/Lyken17/Efficient-PyTorch/ and simplified. Fixed quite a few warnings and made it easier to use via commandline. Tested on both Windows and Linux systems using Python 3.8.

# Speed overview
Trained on the Cats versus Dogs dataset avaliable on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). Results compare the torch.ImageFolder and our lmdb implementation. These are the results using a local SSD:

```
Timings for lmdb (my own implementation)
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
These are the results using a network file system (NFS) drive:

```
Timings for lmdb (my own implementation)
Avg data time: 0.040608997247657
Avg batch time: 0.06778134983413074
Total data time: 7.9593634605407715
Total batch time: 13.285144567489624

Timings for imagefolder: 
Avg data time: 0.056209570291090985
Avg batch time: 0.08088788086054277
Total data time: 11.017075777053833
Total batch time: 15.854024648666382
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
The [converter](folder2lmdb.py) can convert a default image-label structure to an LMDB file (see above). For example, to run it on Linux, given the Dogs vs Cats dataset is in /data and it has a subfolder called "train":

```bash
python folder2lmdb.py -f ~/pytorch-lmdb/data/cats_vs_dogs -s "train"
```

## ImageFolderLMDB
The usage of `ImageFolderLMDB` is identical to `torchvision.datasets`. 

```python
import ImageFolderLMDB
from torch.utils.data import DataLoader
dst = ImageFolderLMDB(path, transform, target_transform)
loader = DataLoader(dst, batch_size=64)
```

## Run the test tool
The [test tool](main.py) takes an ImageFolder path and a LMDB database path, runs training on the Dogs vs Cats dataset and output the results in terms of timings. For example, to run it on Linux, given the Dogs vs Cats dataset is in /data and the already created LMDB file is too:

```bash
python main.py -f ~/pytorch-lmdb/data/cats_vs_dogs/train -l ~/pytorch-lmdb/data/cats_vs_dogs/train.lmdb
```
