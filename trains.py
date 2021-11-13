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
    if stepiters < 300:
        lr = 0.0001
    elif stepiters < 2000:
        lr = 0.001
    elif stepiters < 3000:
        lr = 0.001
    elif epoch < 330:
        lr = 0.001
    elif epoch < 360:
        lr = 0.0001
    else:
        import sys
        sys.exit(0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def trainer():
    config = {'a_bit':8, 'w_bit':8, "all_positive":False, "per_channel":True, 
              "num_classes":10,"batch_init":20}
    pretrainedmodel = r''
    scratch = True #从最开始训练，不是finetuning， 若=False就是finetuning
    tim = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')
    logfile = r'log\log_%s.txt'%tim
    flogs = open(logfile, 'w')

    transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.Resize((32, 32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 3

    trainset = torchvision.datasets.CIFAR10(root='datas', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='datas', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_epochs = 361
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained=True, num_classes=config['num_classes'])

    Floatmodel = True    #QAT or float-32 train
    LSQplus = True     #LSQ+ or LSQ

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

    print(model)
    flogs.write(str(model)+'\n')
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

    adam = False
    batch_size = 3
    lr = 0.001 # initial learning rate (SGD=1E-2, Adam=1E-3)
    momnetum=0.97
    params = [p for p in model.parameters() if p.requires_grad]
    if adam:
        optimizer = optim.Adam(params, lr=lr, betas=(momnetum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=momnetum, nesterov=True)
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
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        flogs.write('Epoch {}/{}'.format(epoch, num_epochs)+'\n')
        print('-'*10)
        running_loss = 0
        if epoch<nowepoch:
            stepiters += len(dataloader) 
            continue
        model.train()
        count = 0
        print("length trainloader is: ", len(trainloader))
        for i, (image, label) in enumerate(trainloader):
            stepiters += 1
            if stepiters<alliters:
                continue
            count += 1
            adjust_lr(optimizer, stepiters, epoch) #
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            epoch_loss = running_loss / count
            logword = 'epoch: {}, iteration: {}, alliters: {}, lr: {}, loss: {:.3f}, avgloss: {:.3f}'.format(
                epoch, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], loss.item(), epoch_loss)
            print(logword)
            flogs.write(logword+'\n')
            flogs.flush()
            savestate = {'state_dict':model.state_dict(),\
                        'iteration':i,\
                        'alliters':stepiters,\
                        'nowepoch':epoch}
            # if stepiters%500==0 and count!=1:
            #     try:
            #         torch.save(savestate, r'.\log\model_{}_{}_{:.3f}_{}.pth'.format(epoch, stepiters, loss.item(),tim))
            #     except:
                    # pass
        print('validation of testes')
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                count+=1
                images, labels = data
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                        accuracy))

        # lr_scheduler.step()
        iteration=0
        try:
            torch.save(savestate, r'.\log\model_{}_{}_{:.3f}_{}.pth'.format(epoch, stepiters, loss.item(),tim))
        except:
            pass
        # evaluate(model, dataloader_test, device = device)
    timeused  = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
    flogs.close()

if __name__ == '__main__':
    trainer()
