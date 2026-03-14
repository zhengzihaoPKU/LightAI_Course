import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
import numpy as np
from torchinfo import summary
from models.simplecnn import SimpleCNN
from models.fc_complex import FC_complex
from models.fc_simple import FC_simple
from models.resnet import*

def parse_args():
    """
    parameters of training
    """
    parser = argparse.ArgumentParser(prog='net train', description='train KD')
    parser.add_argument('--dataset', type=str, default='cifar10', help='the dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='training device')
    parser.add_argument('--seed', type=int, default=0, help='training seed')
    parser.add_argument('--save_path', type=str, default='Prune_log/1.txt', help='training seed')
    parser.add_argument('--batch_size', type=int, default=128, help='the batchsize of teacher model')
    parser.add_argument('--img_size', type=int, default=32, help='the image size of dataset')
    parser.add_argument('--in_channels', type=int, default=3, help='the in channels of dataset')
    parser.add_argument('--num_classes', type=int, default=32, help='the in channels of dataset')
    parser.add_argument('--load', type=int, default=0, help='load exist model')
    parser.add_argument('--model', type=str, default='FC_complex', help='the teacher model')
    parser.add_argument('--lr', type=float, default=1e-4, help='the lr of teacher model')
    parser.add_argument('--optimizer', type=str, default='Adam', help='the optimizer of teacher model')
    parser.add_argument('--epoch', type=int, default=5, help='the epoch of teacher model')
    parser.add_argument('--prune_type', type=str, default='remove', help='the type of pruning')
    parser.add_argument('--prune_rate', type=float, default=0.5, help='the rate of pruning')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-4, help='the rate of pruning')
    parser.add_argument('--fine_tune_epoch', type=int, default=10, help='the rate of pruning')
    args = parser.parse_args()
    return vars(args)

def random_seed(config):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_dataloader(config):
    if config['dataset'] == 'mnist':
        train_dataset=torchvision.datasets.MNIST(root="/rshome/maoliang.li/dataset",train=True,transform=transforms.ToTensor(),download=True)
        test_dateset=torchvision.datasets.MNIST(root="/rshome/maoliang.li/dataset",train=False,transform=transforms.ToTensor(),download=True)
        config['img_size'] = 28
        config['in_channels'] = 1
        config['num_classes'] = 10
    elif config['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset=torchvision.datasets.CIFAR10(root="/rshome/maoliang.li/dataset/cifar10",train=True,transform=transform_train,download=True)
        test_dateset=torchvision.datasets.CIFAR10(root="/rshome/maoliang.li/dataset/cifar10",train=False,transform=transform_test,download=True)
        config['img_size'] = 32
        config['in_channels'] = 3
        config['num_classes'] = 10
    elif config['dataset'] == 'cifar100':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset=torchvision.datasets.CIFAR100(root="/rshome/maoliang.li/dataset/cifar100",train=True,transform=transform_train,download=True)
        test_dateset=torchvision.datasets.CIFAR100(root="/rshome/maoliang.li/dataset/cifar100",train=False,transform=transform_test,download=True)
        config['img_size'] = 32
        config['in_channels'] = 3
        config['num_classes'] = 100
    train_dataloder=DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_dataloder=DataLoader(test_dateset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    return train_dataloder, test_dataloder

def get_network(config):
    if config['model'] == 'FC_complex':
        model = FC_complex(config['in_channels'], config['img_size'], config['num_classes'])
    elif config['model'] == 'resnet101':
        model = ResNet101(num_classes=config['num_classes'])
    elif config['model'] == 'resnet50':
        model = ResNet50(num_classes=config['num_classes'])
    elif config['model'] == 'resnet18':
        model = ResNet18(num_classes=config['num_classes'])
    elif config['model'] == 'simplecnn':
        model = SimpleCNN(config['in_channels'], config['img_size'], config['num_classes'])
    return model

def get_optim(config, model):
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    return optimizer 

def train_model(config, train_dataloader, test_dataloader, optimizer, model, loss_function, scheduler):
    for epoch in range(config['epoch']):
        print("epoches:{}, lr:{:.6f}".format(epoch,scheduler.get_last_lr()[0]))
        model.train()
        for image, label in train_dataloader:
            image, label = image.to(device),label.to(device)
            optimizer.zero_grad()
            out=model(image)
            loss=loss_function(out,label)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        num_correct=0
        num_samples=0
        with torch.no_grad():
            for image,label in test_dataloader:
                image=image.to(device)
                label=label.to(device)
                out=model(image)
                pre=out.max(1).indices
                num_correct+=(pre==label).sum()
                num_samples+=pre.size(0)
            acc=(num_correct/num_samples).item()
        model.train()
        print("accurate={:.4f}".format(acc))
    torch.save(model, 'model.pt')

def save_config(config):
    with open(config['save_path'], 'w') as f:
        for k,v in config.items():
            print(f'{k}  ||  {v}', file=f, flush=True)

def model_prune_remove(config, model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    print('sparsity of model after pruning: ')
    print(count_sparsity(model))

def model_prune(config, model):
    if config['prune_type'] == 'unstructured':
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=config['prune_rate'])
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=config['prune_rate'])
    elif config['prune_type'] == 'structured':
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=config['prune_rate'], n=2, dim=0)
            if isinstance(module, torch.nn.Linear):
                prune.ln_structured(module, name='weight', amount=config['prune_rate'], n=2, dim=0)

def model_fine_tune(config, model):
    fine_tune_optimizer = torch.optim.Adam(model.parameters(),lr=config['fine_tune_lr'])
    fine_tune_loss_function = nn.CrossEntropyLoss()
    for epoch in range(config['fine_tune_epoch']):
        print("epoches:{}, lr:{:.6f}".format(epoch, config['fine_tune_lr']))
        model.train()
        for image, label in train_dataloader:
            image, label = image.to(device),label.to(device)
            optimizer.zero_grad()
            out=model(image)
            loss=fine_tune_loss_function(out,label)
            loss.backward()
            fine_tune_optimizer.step()
        model.eval()
        num_correct=0
        num_samples=0
        with torch.no_grad():
            for image,label in test_dataloader:
                image=image.to(device)
                label=label.to(device)
                out=model(image)
                pre=out.max(1).indices
                num_correct+=(pre==label).sum()
                num_samples+=pre.size(0)
            acc=(num_correct/num_samples).item()
        print("accurate={:.4f}".format(acc))
        print('sparsity of model during fine_tune: ')
        print(count_sparsity(model))
        model.train()
    torch.save(model, 'model_prune.pt')

def count_sparsity(model):
    sparsity = 0
    num = 0
    for k,v in model.state_dict().items():
        if 'weight' in k:
            num += v.numel()
            sparsity += (v==0).sum().item()
    return sparsity / num

if __name__ == '__main__':
    config = parse_args()
    save_config(config)
    #设置随机种子
    random_seed(config)
    train_dataloader, test_dataloader = get_dataloader(config)
    device = config['device']
    model= get_network(config)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = get_optim(config, model)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epoch'], eta_min=1e-9)
    if config['load'] == 1:
        print('****************Load exist model...******************')
        model=torch.load('model.pt')
    else:
        print('****************Start to train...******************')
        train_model(config, train_dataloader, test_dataloader, optimizer, model, loss_function, scheduler)
    print('sparsity of model before: ')
    print(count_sparsity(model))
    print('****************Start to fine-tune...******************')
    model_prune(config, model)
    model_fine_tune(config, model)
    # remove must after fine-tune
    model_prune_remove(config, model)


# # 1. 权重剪枝
# prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# # 2. 结构化剪枝
# prune.ln_structured(model.conv1, name='weight', amount=0.5, n=2, dim=0)

# # 3. 全局剪枝
# parameters_to_prune = (
#     (model.conv1, 'weight'),
#     (model.conv2, 'weight'),
#     (model.fc1, 'weight'),
# )
# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.2
# )

# # 4. 移除剪枝
# for module in model.modules():
#     if isinstance(module, torch.nn.Conv2d):
#         prune.remove(module, 'weight')