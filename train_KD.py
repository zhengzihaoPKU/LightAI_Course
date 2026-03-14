import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument('--save_path', type=str, default='log/1.txt', help='training seed')
    parser.add_argument('--batch_size', type=int, default=128, help='the batchsize of teacher model')
    parser.add_argument('--img_size', type=int, default=32, help='the image size of dataset')
    parser.add_argument('--in_channels', type=int, default=3, help='the in channels of dataset')
    parser.add_argument('--num_classes', type=int, default=32, help='the in channels of dataset')

    parser.add_argument('--teacher_model', type=str, default='FC_complex', help='the teacher model')
    parser.add_argument('--teacher_lr', type=float, default=1e-4, help='the lr of teacher model')
    parser.add_argument('--teacher_optimizer', type=str, default='Adam', help='the optimizer of teacher model')
    parser.add_argument('--teacher_epoch', type=int, default=5, help='the epoch of teacher model')
    parser.add_argument('--load', type=int, default=0, help='the epoch of teacher model')

    parser.add_argument('--student_model', type=str, default='FC_simple', help='the student model')
    parser.add_argument('--student_lr', type=float, default=1e-4, help='the lr of student model')
    parser.add_argument('--student_optimizer', type=str, default='Adam', help='the optimizer of student model')
    parser.add_argument('--student_epoch', type=int, default=5, help='the epoch of student model')
    
    parser.add_argument('--kd_model', type=str, default='FC_simple', help='the KD model')
    parser.add_argument('--kd_lr', type=float, default=1e-4, help='the lr of KD model')
    parser.add_argument('--kd_optimizer', type=str, default='Adam', help='the optimizer of KD model')
    parser.add_argument('--kd_epoch', type=int, default=5, help='the epoch of KD model')
    parser.add_argument('--kd_T', type=float, default=7, help='the T of KD model')
    parser.add_argument('--kd_alpha', type=float, default=0.2, help='the epoch of KD model')
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
    if config['teacher_model'] == 'FC_complex':
        teacher_model = FC_complex(config['in_channels'], config['img_size'], config['num_classes'])
    elif config['teacher_model'] == 'resnet101':
        teacher_model = ResNet101(num_classes=config['num_classes'])
    elif config['teacher_model'] == 'resnet50':
        teacher_model = ResNet50(num_classes=config['num_classes'])
    elif config['teacher_model'] == 'resnet18':
        teacher_model = ResNet18(num_classes=config['num_classes'])
    elif config['teacher_model'] == 'simplecnn':
        teacher_model = SimpleCNN(config['in_channels'], config['img_size'], config['num_classes'])


    if config['student_model'] == 'FC_simple':
        student_model = FC_simple(config['in_channels'], config['img_size'], config['num_classes'])
    elif config['student_model'] == 'resnet18':
        student_model = ResNet18(num_classes=config['num_classes'])
    elif config['student_model'] == 'simplecnn':
        student_model = SimpleCNN(config['in_channels'], config['img_size'], config['num_classes'])

    if config['kd_model'] == 'FC_simple':
        kd_model = FC_simple(config['in_channels'], config['img_size'], config['num_classes'])
    elif config['kd_model'] == 'resnet18':
        kd_model = ResNet18(num_classes=config['num_classes'])
    elif config['kd_model'] == 'simplecnn':
        kd_model = SimpleCNN(config['in_channels'], config['img_size'], config['num_classes'])
    return teacher_model, student_model, kd_model

def get_optim(config, teacher_model, student_model, kd_model):
    if config['teacher_optimizer'] == 'Adam':
        teacher_optimizer = torch.optim.Adam(teacher_model.parameters(),lr=config['teacher_lr'])
    if config['student_optimizer'] == 'Adam':
        student_optimizer = torch.optim.Adam(student_model.parameters(),lr=config['student_lr'])
    if config['kd_optimizer'] == 'Adam':
        kd_optimizer = torch.optim.Adam(kd_model.parameters(),lr=config['kd_lr'])
    return teacher_optimizer, student_optimizer, kd_optimizer


def train_teacher_model(config, train_dataloader, test_dataloader, teacher_optimizer, teacher_model, teacher_loss_function, teacher_scheduler):
    for epoch in range(config['teacher_epoch']):
        print("epoches:{}, lr:{:.6f}".format(epoch,teacher_scheduler.get_last_lr()[0]))
        teacher_model.train()
        for image, label in train_dataloader:
            image, label = image.to(device),label.to(device)
            teacher_optimizer.zero_grad()
            out=teacher_model(image)
            loss=teacher_loss_function(out,label)
            loss.backward()
            teacher_optimizer.step()
        teacher_scheduler.step()
        teacher_model.eval()
        num_correct=0
        num_samples=0
        with torch.no_grad():
            for image,label in test_dataloader:
                image=image.to(device)
                label=label.to(device)
                out=teacher_model(image)
                pre=out.max(1).indices
                num_correct+=(pre==label).sum()
                num_samples+=pre.size(0)
            acc=(num_correct/num_samples).item()
        teacher_model.train()
        print("accurate={:.4f}".format(acc))
    torch.save(teacher_model, 'teacher_model.pt')

def train_student_model(config, train_dataloader, test_dataloader, student_optimizer, student_model, student_loss_function, student_scheduler):
    for epoch in range(config['student_epoch']):
        print("epoches:{}, lr:{:.6f}".format(epoch,student_scheduler.get_last_lr()[0]))
        student_model.train()
        for image, label in train_dataloader:
            image, label = image.to(device),label.to(device)
            student_optimizer.zero_grad()
            out=student_model(image)
            loss=student_loss_function(out,label)
            loss.backward()
            student_optimizer.step()
        student_scheduler.step()
        student_model.eval()
        num_correct=0
        num_samples=0
        with torch.no_grad():
            for image,label in test_dataloader:
                image=image.to(device)
                label=label.to(device)
                out=student_model(image)
                pre=out.max(1).indices
                num_correct+=(pre==label).sum()
                num_samples+=pre.size(0)
            acc=(num_correct/num_samples).item()
        student_model.train()
        print("accurate={:.4f}".format(acc))

def train_kd_model(config, train_dataloader, test_dataloader, teacher_model, kd_model, kd_optimizer, kd_hardloss_function, kd_softloss_function, kd_scheduler):
    teacher_model.eval()
    T = config['kd_T']
    alpha = config['kd_alpha']
    for epoch in range(config['kd_epoch']):
        print("epoches:{}, lr:{:.6f}".format(epoch,kd_scheduler.get_last_lr()[0]))
        kd_model.train()
        for image,label in train_dataloader:
            image,label=image.to(device),label.to(device)
            with torch.no_grad():
                teacher_output=teacher_model(image)
            kd_optimizer.zero_grad()
            out=kd_model(image)
            loss=kd_hardloss_function(out,label)
            ditillation_loss=F.kl_div(F.softmax(out/T,dim=1),F.softmax(teacher_output/T,dim=1), reduction='batchmean')*T*T
            loss_all=loss*alpha+ditillation_loss*(1-alpha)
            loss_all.backward()
            kd_optimizer.step()
        kd_scheduler.step()
        kd_model.eval()
        num_correct=0
        num_samples=0
        with torch.no_grad():
            for image,label in test_dataloader:
                image=image.to(device)
                label=label.to(device)
                out=kd_model(image)
                pre=out.max(1).indices
                num_correct+=(pre==label).sum()
                num_samples+=pre.size(0)
            acc=(num_correct/num_samples).item()
        kd_model.train()
        print("accurate={:.4f}".format(acc))

def save_config(config):
    with open(config['save_path'], 'w') as f:
        for k,v in config.items():
            print(f'{k}  ||  {v}', file=f, flush=True)

if __name__ == '__main__':
    config = parse_args()
    save_config(config)
    #设置随机种子
    random_seed(config)
    train_dataloader, test_dataloader = get_dataloader(config)
    device = config['device']
    teacher_model, student_model, kd_model = get_network(config)
    teacher_model.to(device)
    student_model.to(device)
    kd_model.to(device)
    teacher_loss_function = nn.CrossEntropyLoss()
    student_loss_function = nn.CrossEntropyLoss()
    kd_hardloss_function = nn.CrossEntropyLoss()
    kd_softloss_function = 1
    teacher_optimizer, student_optimizer, kd_optimizer = get_optim(config, teacher_model, student_model, kd_model)
    teacher_scheduler = lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=config['teacher_epoch'], eta_min=1e-9)
    student_scheduler = lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=config['student_epoch'], eta_min=1e-9)
    kd_scheduler = lr_scheduler.CosineAnnealingLR(kd_optimizer, T_max=config['kd_epoch'], eta_min=1e-9)
    print('****************Start to train teacher model...******************')
    if config['load'] == 1:
        teacher_model = torch.load('model.pt').to(device)
    else:
        train_teacher_model(config, train_dataloader, test_dataloader, teacher_optimizer, teacher_model, teacher_loss_function, teacher_scheduler)
    print('****************Start to train student model...******************')
    train_student_model(config, train_dataloader, test_dataloader, student_optimizer, student_model, student_loss_function, student_scheduler)
    print('******************Start to train KD model...*********************')
    train_kd_model(config, train_dataloader, test_dataloader, teacher_model, kd_model, kd_optimizer, kd_hardloss_function, kd_softloss_function, kd_scheduler)
