import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import normalize

from config import config
from datasets import CategoriesSampler, DataSet
from models.ici import ICI
from models.baseline import RandomPick
from models.consistancy import FixmatchPick
from utils import get_embedding, mean_confidence_interval, setup_seed


def train_embedding(args):
    setup_seed(2333)
    ckpt_root = os.path.join('./ckpt', args.dataset)
    os.makedirs(ckpt_root, exist_ok=True)
    if args.dataset == 'miniimagenet':
        data_root = os.path.join(args.folder, 'miniimagenet/images-lc/')
    elif args.dataset == 'tieredimagenet':
        data_root = os.path.join(args.folder, 'tieredImageNet/')
    else:
        data_root = os.path.join(args.folder, args.dataset)
    from datasets import EmbeddingDataset
    source_set = EmbeddingDataset(data_root, args.img_size, 'train')
    source_loader = DataLoader(
        source_set, num_workers=args.num_workers, batch_size=64, shuffle=True)
    # test_set = EmbeddingDataset(data_root, args.img_size, 'val')
    test_set = EmbeddingDataset(data_root, args.img_size, 'trainval')
    test_loader = DataLoader(test_set, num_workers=args.num_workers, batch_size=32, shuffle=False)

    if args.dataset == 'cub':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    else:
        num_classes = 64
    from models.resnet12 import resnet12
    model = resnet12(num_classes).to(args.device)
    model = model.to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(120):
        model.train()
        scheduler.step(epoch)
        loss_list = []
        train_acc_list = []
        for images, labels in tqdm(source_loader, ncols=0):
            preds = model(images.to(args.device))
            loss = criterion(preds, labels.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            train_acc_list.append(preds.max(1)[1].cpu().eq(
                labels).float().mean().item())
        acc = []
        model.eval()
        for images, labels in test_loader:
            preds = model(images.to(args.device)).detach().cpu()
            preds = torch.argmax(preds, 1).reshape(-1)
            labels = labels.reshape(-1)
            acc += (preds==labels).tolist()
        acc = np.mean(acc)
        print('Epoch:{} Train-loss:{} Train-acc:{} Valid-acc:{}'.format(epoch, str(np.mean(loss_list))[:6], str(
            np.mean(train_acc_list))[:6], str(acc)[:6]))
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(
                ckpt_root, "res12_epoch{}.pth.tar".format(epoch))
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), os.path.join(ckpt_root,'res12_best.pth.tar'))

def test(args):
    setup_seed(2333)
    import warnings
    warnings.filterwarnings('ignore')
    if args.dataset == 'cub':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    else:
        num_classes = 64

    if args.resume is not None:
        from models.resnet12 import resnet12
        model = resnet12(num_classes).to(args.device)
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = nn.DataParallel(model)
    model.to(args.device)
    model.eval()

    if args.data_picker == "ici":
        data_picker = ICI(classifier=args.classifier, num_class=args.num_test_ways,
                step=args.step, reduce=args.embed, d=args.dim)
        dataset = DataSet(data_root, 'test', args.img_size)

    elif args.data_picker == "random":
        data_picker = RandomPick(classifier=args.classifier, num_class=args.num_test_ways,
                step=args.step, reduce=args.embed, d=args.dim)
        dataset = DataSet(data_root, 'test', args.img_size)

    elif args.data_picker == "fixmatch":
        data_picker = FixmatchPick(args.img_size, classifier=args.classifier, num_class=args.num_test_ways,
                step=args.step, reduce=args.embed, d=args.dim)
        # dataset = FixmatchDataset(data_root, 'test', args.img_size)
        dataset = DataSet(data_root, 'test', args.img_size)
    else:    
        raise NotImplementedError

    if args.dataset == 'miniimagenet':
        data_root = os.path.join(args.folder, 'miniimagenet/images-lc/')
    elif args.dataset == 'tieredimagenet':
        data_root = os.path.join(args.folder, 'tieredImageNet/')
    else:
        data_root = os.path.join(args.folder, args.dataset)
    
    # dataset = DataSet(data_root, 'val', args.img_size)

    sampler = CategoriesSampler(dataset.label, args.num_batches,
                                args.num_test_ways, (args.num_shots, 15, args.unlabel))
    testloader = DataLoader(dataset, batch_sampler=sampler,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    k = args.num_shots * args.num_test_ways
    loader = tqdm(testloader, ncols=0)
    iterations = math.ceil(args.unlabel/args.step) + \
        2 if args.unlabel != 0 else math.ceil(15/args.step) + 2
    acc_list = [[] for _ in range(iterations)]

    for data, indicator in loader:
        targets = torch.arange(args.num_test_ways).repeat(args.num_shots+15+args.unlabel).long()[
            indicator[:args.num_test_ways*(args.num_shots+15+args.unlabel)] != 0]
        data = data[indicator != 0].to(args.device)
        train_inputs = data[:k]
        train_targets = targets[:k].cpu().numpy()
        test_inputs = data[k:k+15*args.num_test_ways]
        test_targets = targets[k:k+15*args.num_test_ways].cpu().numpy()
        # loader.set_postfix_str("Data loading ready")
        train_embeddings = get_embedding(model, train_inputs, args.device)
        # loader.set_postfix_str("train_embedding ready")
        data_picker.fit(train_embeddings, train_targets)
        # loader.set_postfix_str("ici complete")
        test_embeddings = get_embedding(model, test_inputs, args.device)
        # loader.set_postfix_str("train_embedding ready")
        if args.unlabel != 0:
            unlabel_inputs = data[k+15*args.num_test_ways:]
            unlabel_embeddings = get_embedding(
                model, unlabel_inputs, args.device)
            # loader.set_postfix_str("unlabel_embedding ready")
        else:
            unlabel_embeddings = None
        acc = data_picker.predict(test_embeddings, unlabel_embeddings,
                          True, test_targets)
        loader.set_postfix({"Mean Acc": np.mean(acc)})
        for i in range(min(iterations-1,len(acc))):
            acc_list[i].append(acc[i])
        acc_list[-1].append(acc[-1])
    mean_list = []
    ci_list = []
    for item in acc_list:
        mean, ci = mean_confidence_interval(item)
        mean_list.append(mean)
        ci_list.append(ci)
    print("Test Acc Mean{}".format(
        ' '.join([str(i*100)[:5] for i in mean_list])))
    print("Test Acc ci{}".format(' '.join([str(i*100)[:5] for i in ci_list])))


def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    if args.mode == 'train':
        train_embedding(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise NameError


if __name__ == '__main__':
    args = config()
    main(args)
