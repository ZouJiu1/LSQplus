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
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def adjust_lr(optimizer, stepiters, epoch):
    # if stepiters < 100: #2warmup start
    #     lr = stepiters*0.01/100
    # elif stepiters < 2000:
    #     lr = 0.001
    # elif stepiters < 3000:
    #     lr = 0.001
    if epoch <= 31:
        lr = 0.1
    elif epoch <= 61:
        lr = 0.01
    elif epoch <= 81:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def trainer():
    #batch_init 使用预训练模型对量化参数进行初始化的iters or steps
    config = {'a_bit':8, 'w_bit':8, "all_positive":False, "per_channel":False, 
              "num_classes":10,"batch_init":20}
    pretrainedmodel = r'C:\Users\10696\Desktop\QAT\lsq+\log\model_108_42510_0.003_92.528_2021-11-27_17-49-47.pth'
    # Resnet_pretrain = False
    batch_size = 128
    num_epochs = 112
    Floatmodel = False    #QAT or float-32 train   False or True
    LSQplus = False       #LSQ+ or LSQ    True or False
    version = 'V1'
    scratch = True       #从最开始训练，不是finetuning， 若=False就是finetuning
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
    # print(torch.__version__)
    time.sleep(3)
    adam = False
    lr = 0.001 # initial learning rate (SGD=1E-2, Adam=1E-3)
    momnetum=0.9
    params = [p for p in model.parameters() if p.requires_grad]
    # if adam:
    #     optimizer = optim.Adam(params, lr=lr, betas=(momnetum, 0.999))  # adjust beta1 to momentum
    # else:
    optimizer = optim.SGD(params, lr=lr, momentum=momnetum, weight_decay=5e-4)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=7,
    #                                                gamma=0.1)
    torch.manual_seed(999999)
    start = time.time()
    print('Using {} device'.format(device))
    flogs.write('Using {} device'.format(device)+'\n')
    stepiters = 0
    criterion = torch.nn.CrossEntropyLoss()
    pre = -999999
    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs))
        flogs.write('Epoch {}/{}'.format(epoch, num_epochs)+'\n')
        print('-'*100)
        running_loss = 0
        if epoch<nowepoch:
            stepiters += len(trainloader)
            continue
        model.train()
        count = 0
        print("length trainloader is: ", len(trainloader))
        train_acc = 0
        train_all = 0
        for i, (image, label) in enumerate(trainloader):
            stepiters += 1
            if stepiters<alliters:
                continue
            count += 1
            lr = adjust_lr(optimizer, stepiters, epoch) #
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            _, predict = torch.max(outputs, 1)
            train_acc += (predict==label).sum()
            train_all += len(label)
            train_Acc = train_acc/train_all

            loss = criterion(outputs, label)
            loss.backward()

            #LSQplus V1论文原版的实现，在前几个的iters使用MSE公式update其s和beta
            if LSQplus and version=='V1' and not Floatmodel and stepiters<plusV1_inititers and epoch==0:
                print(stepiters, ': update_LSQplus_activation_Scalebeta')
                model = update_LSQplus_activation_Scalebeta(model)
            optimizer.step()
            # statistics
            running_loss += loss.item()
            epoch_loss = running_loss / count
            logword = 'epoch: {}, iteration: {}, alliters: {}, lr: {}, loss: {:.3f}, avgloss: {:.3f}, train_Acc: {:.3f}'.format(
                epoch, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], loss.item(), epoch_loss, train_Acc)
            if i%showstep==0:
                print(logword)
                flogs.write(logword+'\n')
                flogs.flush()
            savestate = {'state_dict':model.state_dict(),\
                        'iteration':i,\
                        'alliters':stepiters,\
                        "lr":lr,\
                        'nowepoch':epoch}
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        if epoch%3==0 and epoch>nowepoch:
            print('validation of testes')
            with torch.no_grad():
                count = 0
                print('length of testloader: ', len(testloader))
                for data in testloader:
                    count += 1
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    # if count==100:
                    #     break
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
                print("Validation Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                            accuracy))
                correctall += correct_count
                alltest += total_pred[classname]
                flogs.write("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy)+'\n')
            flogs.flush()
            Accuracy = round(100 * float(correctall)/alltest, 3)
            print("Accuracy all is: {:.1f}".format(Accuracy))

            # lr_scheduler.step()
            iteration=0
            try:
                if epoch>nowepoch and Accuracy>pre:
                    torch.save(savestate, os.path.join(savepath, prefix+'_models_{}_{}_{}_{:.3f}_{}_{}.pth'.format(
                        lr, epoch, stepiters, loss.item(),Accuracy,tim)))
                pre = Accuracy
            except:
                pass
        # evaluate(model, dataloader_test, device = device)
    timeused  = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
    flogs.close()

if __name__ == '__main__':
    trainer()
