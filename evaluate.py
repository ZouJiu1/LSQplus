#encoding=utf-8
#Author: ZouJiu
#Time: 2021-11-13

import numpy as np
import torch
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from load_datas import TF, trainDataset, collate_fn
import models #, resnet50
from quantization.lsqquantize import prepare as lsqprepare
from quantization.lsqplus_quantize import prepare as lsqplusprepare
import torch.optim as optim
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def adjust_lr(optimizer, stepiters, epoch):
    if epoch < 135:
        lr = 0.1
    elif epoch < 185:
        lr = 0.01
    elif epoch < 290:
        lr = 0.001
    else:
        import sys
        sys.exit(0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate():
    config = {'a_bit':8, 'w_bit':8, "all_positive":False, "per_channel":True, 
              "num_classes":10,"batch_init":20}
    pretrainedmodel = r''
    Resnet_pretrain = False #test
    batch_size = 128
    num_epochs = 290
    Floatmodel = True    #QAT or float-32 train
    LSQplus = True     #LSQ+ or LSQ
    scratch = True #从最开始训练，不是finetuning， 若=False就是finetuning
    tim = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.ToTensor()])

    batch_size = 128 #Accuracy all is: 73.4

    testset = torchvision.datasets.CIFAR10(root='datas', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained = Resnet_pretrain, num_classes=config['num_classes'])

    #LSQ+
    if LSQplus and not Floatmodel:
        lsqplusprepare(model, inplace=True, a_bits=config["a_bit"], w_bits=config["w_bit"],
                all_positive=config["all_positive"], per_channel=config["per_channel"],
                batch_init = config["batch_init"])
    elif not LSQplus and not Floatmodel:
        #LSQ
        lsqprepare(model, inplace=True, a_bits=config["a_bit"], w_bits=config["w_bit"],
                all_positive=config["all_positive"], per_channel=config["per_channel"],
                batch_init = config["batch_init"])
    elif Floatmodel:
        pass

    if not Floatmodel:
        print(model)
    if not os.path.exists(pretrainedmodel):
        print('the pretrainedmodel do not exists %s'%pretrainedmodel)
    if pretrainedmodel and os.path.exists(pretrainedmodel):
        print('loading pretrained model: ', pretrainedmodel)
        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location='cuda')
        else:
            state_dict = torch.load(pretrainedmodel, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
        if not scratch:
            iteration = state_dict['iteration']
            alliters = state_dict['alliters']
            nowepoch = state_dict['nowepoch']
        else:
            iteration = 0
            alliters = 0
            nowepoch = 0
        print('loading complete')
    else:
        print('no pretrained model')
        iteration = 0
        alliters = 0
        nowepoch = 0
    model = model.to(device)

    print('validation of testes')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    correctall = 0
    alltest = 0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
        correctall += correct_count
        alltest += total_pred[classname]
    print("Accuracy all is: {:.1f}".format(100 * float(correctall)/alltest))
    

if __name__ == '__main__':
    evaluate()
