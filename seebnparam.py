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
from quantization.lsqquantize_V1 import prepare as lsqprepareV1
from quantization.lsqquantize_V2 import prepare as lsqprepareV2
from quantization.lsqplus_quantize_V1 import prepare as lsqplusprepareV1
from quantization.lsqplus_quantize_V2 import prepare as lsqplusprepareV2
from quantization.lsqplus_quantize_V1 import update_LSQplus_activation_Scalebeta
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def adjust_lr(optimizer, stepiters, epoch):
    # if stepiters < 100: #2warmup start
    #     lr = stepiters*0.01/100
    # elif stepiters < 2000:
    #     lr = 0.001
    # elif stepiters < 3000:
    #     lr = 0.001
    if epoch <= 30:
        lr = 0.1
    elif epoch <= 46:
        lr = 0.01
    elif epoch <= 55:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def trainer():
    #batch_init 使用预训练模型对量化参数进行初始化的iters or steps
    config = {'a_bit':8, 'w_bit':8, "all_positive":False, "per_channel":True, 
              "num_classes":10,"batch_init":20}
    pretrainedmodel = r'C:\Users\10696\Desktop\QAT\lsq+\log\model_108_42510_0.003_92.528_2021-11-27_17-49-47.pth'
    # Resnet_pretrain = False
    batch_size = 128
    num_epochs = 112
    Floatmodel = True    #QAT or float-32 train   False or True
    LSQplus = False       #LSQ+ or LSQ    True or False
    version = 'V1'
    scratch = False       #从最开始训练，不是finetuning， 若=False就是finetuning
    showstep = 31
    #LSQPlusActivationQuantizer里的self.beta初始值要关注
    plusV1_inititers = 30 #update激活层的量化参数s和beta
    assert showstep > 0
    assert isinstance(showstep, int)
    assert isinstance(batch_size, int)
    assert isinstance(num_epochs, int)
    if Floatmodel:
        prefix = 'float32'
    elif LSQplus and not Floatmodel and version=='V1':
        if  not config['per_channel']:
            prefix = 'LSQplus_V1'
        else:
            prefix = 'LSQplus_V1_pcl'
    elif LSQplus and not Floatmodel and version=='V2':
        if  not config['per_channel']:
            prefix = 'LSQplus_V2'
        else:
            prefix = 'LSQplus_V2_pcl'
    elif not LSQplus and not Floatmodel and version=='V1':
        if  not config['per_channel']:
            prefix = 'LSQ_V1'
        else:
            prefix = 'LSQ_V1_pcl'
    elif not LSQplus and not Floatmodel and version=='V2':
        if  not config['per_channel']:
            prefix = 'LSQ_V2'
        else:
            prefix = 'LSQ_V2_pcl'
    else:
        print('setting is wrong......, please check it')
        exit(-1)

    tim = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')
    logfile = r'log'+os.sep+prefix+'_log_%s.txt'%tim
    savepath = r'log'
    flogs = open(logfile, 'w')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
    test_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    trainset = torchvision.datasets.CIFAR10(root='datas', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='datas', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(num_classes=config['num_classes'])

    #LSQ+
    if LSQplus and not Floatmodel and version=='V1':
        #LSQplus V1
        lsqplusprepareV1(model, inplace=True, a_bits=config["a_bit"], w_bits=config["w_bit"],
                all_positive=config["all_positive"], per_channel=config["per_channel"],
                batch_init = config["batch_init"])
        print(model, '\npreparing lsqplus V1 models')
    elif LSQplus and not Floatmodel and version=='V2':
        #LSQplus V2
        lsqplusprepareV2(model, inplace=True, a_bits=config["a_bit"], w_bits=config["w_bit"],
                all_positive=config["all_positive"], per_channel=config["per_channel"],
                batch_init = config["batch_init"])
        print(model, '\npreparing lsqplus V2 models')
    elif not LSQplus and not Floatmodel and version=='V1':
        #LSQ V1
        lsqprepareV1(model, inplace=True, a_bits=config["a_bit"], w_bits=config["w_bit"],
                all_positive=config["all_positive"], per_channel=config["per_channel"],
                batch_init = config["batch_init"])
        print(model, '\npreparing lsq V1 models')
    elif not LSQplus and not Floatmodel and version=='V2':
        #LSQ V2
        lsqprepareV2(model, inplace=True, a_bits=config["a_bit"], w_bits=config["w_bit"],
                all_positive=config["all_positive"], per_channel=config["per_channel"],
                batch_init = config["batch_init"])
        print(model, '\npreparing lsq V2 models')
    elif Floatmodel:
        print(model, '\npreparing float models')
        pass
    # if not Floatmodel:
        # print(model)
    flogs.write(str(model)+'\n')
    if not os.path.exists(pretrainedmodel):
        print('the pretrainedmodel do not exists %s'%pretrainedmodel)
    if pretrainedmodel and os.path.exists(pretrainedmodel):
        print('loading pretrained model: ', pretrainedmodel)
        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location='cuda')
        else:
            state_dict = torch.load(pretrainedmodel, map_location='cpu')
        missingkeys, unexpected_keys = model.load_state_dict(state_dict['state_dict'], strict=False)
        print('missingkeys: ', missingkeys)
        print('unexpected_keys: ', unexpected_keys)
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

    weight = []
    count = 0
    weightsepa = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            w = m.weight.data.clone().detach().numpy() #out channel, in channel, h, w
            out_channel = w.shape[0]
            w_per_channel = np.reshape(w, (out_channel, -1))
            w_per_layer = np.reshape(w, (-1))
            print(w_per_channel.shape, w_per_layer.shape)
            weight.append(w_per_layer)
            weightsepa.extend(w_per_channel)
            print(len(weightsepa[-1]))
            count += 1
    
    print(len(weightsepa[11]))
    plt.hist(weightsepa[11], bins=100)
    plt.title("all weights parameters")
    plt.ylabel('numbers')
    plt.xlabel("weights")
    plt.show()

    # plt.figure(figsize=(1620,1620))
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axs[i, j].hist(weight[i+j], bins=100)
            # axs[i, j].set_title("weights of layer %d"%(i+j+1))

    plt.show()

    bn = []
    count = 0
    bnsepa = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            gammas = list(m.weight.data.clone().detach().numpy())
            bn.extend(gammas)
            bnsepa.append(gammas)
            print(len(bnsepa[-1]))
            count += 1
    
    plt.hist(bn, bins=100)
    plt.ylabel('numbers')
    plt.xlabel("γ")
    plt.show()

    # bn.sort()
    plt.plot(np.arange(len(bn)), bn)
    plt.title("resnet18 BN γ parameters γ*x+β")
    plt.ylabel("no sorted γ")
    plt.xlabel("indexs")
    plt.show()

    bn.sort()
    plt.plot(np.arange(len(bn)), bn)
    plt.title("resnet18 BN γ parameters γ*x+β")
    plt.ylabel("sorted γ")
    plt.xlabel("indexs")
    plt.show()

    tmp9 = bnsepa[2]
    plt.plot(np.arange(len(tmp9)), tmp9)
    plt.title("resnet18 BN γ parameters γ*x+β")
    plt.ylabel("no sorted γ")
    plt.xlabel("3th layer")
    plt.show()

    tmp9 = bnsepa[2]
    tmp9.sort()
    plt.plot(np.arange(len(tmp9)), tmp9)
    plt.title("resnet18 BN γ parameters γ*x+β")
    plt.ylabel("sorted γ")
    plt.xlabel("3th layer")
    plt.show()

    tmp9 = bnsepa[17]
    plt.plot(np.arange(len(tmp9)), tmp9)
    plt.title("resnet18 BN γ parameters γ*x+β")
    plt.ylabel("no sorted γ")
    plt.xlabel("18th layer")
    plt.show()
    
    tmp9 = bnsepa[17]
    tmp9.sort()
    plt.plot(np.arange(len(tmp9)), tmp9)
    plt.title("resnet18 BN γ parameters γ*x+β")
    plt.ylabel("sorted γ")
    plt.xlabel("18th layer")
    plt.show()

    plt.close()
    print(len(bnsepa))
    print(bn[:30])
    print(bn[-30:])

if __name__ == '__main__':
    trainer()
