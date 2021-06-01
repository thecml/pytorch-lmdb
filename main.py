import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from folder2lmdb import ImageFolderLMDB

DBS = ['lmdb', 'imagefolder']
PRINT_STATUS = False
BATCH_SIZE = 128

def main(imagefolder_data_dir, lmdb_data_db):
    # create model
    model = models.resnet18(pretrained=True)
    model_params = model.parameters()
        
    # send model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = model.to(device)
    
    # define criterion + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_params, lr=0.01)

    # define normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    for dataset_type in DBS:
        if dataset_type == 'lmdb':
            train_dataset = ImageFolderLMDB(
                lmdb_data_db,
                transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train_dataset = torchvision.datasets.ImageFolder(
                imagefolder_data_dir,
                transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True)
    
        batch_time, data_time = train(train_loader, model, criterion, optimizer, device)
        print(f"Timings for {dataset_type}: ")
        print(f"Avg data time: {data_time.avg}")
        print(f"Avg batch time: {batch_time.avg}")
        print(f"Total data time: {data_time.sum}")
        print(f"Total batch time: {batch_time.sum}\n")
    
def train(train_loader, model, criterion, optimizer, device, epoch=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # send input + target to gpu
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target.squeeze())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            if PRINT_STATUS:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            
    return batch_time, data_time

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_values = list()        

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.avg_values.append(self.avg)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--imagefolder_data_dir", type=str)
    parser.add_argument('-l', '--lmdb_data_db', type=str)
    
    args = parser.parse_args()
    
    main(args.imagefolder_data_dir, args.lmdb_data_db)