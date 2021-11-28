# LSQ and LSQ+<br>
LSQ+ or LSQplus and LSQ <br>

I'm not the author, I just complish an unofficial implementation of LSQ+ or LSQplus and LSQ，the origin paper you can find LSQ+ here [arxiv.org/abs/2004.09576](https://arxiv.org/abs/2004.09576) and LSQ here [arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153).<br>

pytorch==1.8.1<br>

You should train 32-bit float model firstly, then you can finetune a low bit-width quantization QAT model by loading the trained 32-bit float model<br>

Dataset used for training is CIFAR10 and model used is Resnet18 revised<br>

## Version introduction
lsqplus_quantize_V1.py: initialize s、beta of activation quantization according to LSQ+[arxiv.org/abs/2004.09576](https://arxiv.org/abs/2004.09576)<br><br>
lsqplus_quantize_V2.py: initialize s、beta of activation quantization according to min max values<br><br>
lsqquantize_V1.py：initialize s of activation quantization according to LSQ [arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153)<br><br>
lsqquantize_V2.py: initialize s of activation quantization = 1<br><br>

## The Train Results 
| version | weight per_channel | learning rate | A s initial | A beta initial | best epoch | Accuracy | models
| ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Float 32bit | - | <=66 0.1<br><=86 0.01<br><=99 0.001<br><=112 0.0001 | - | - | 112 | 92.6 | [download](https://share.weiyun.com/g7P6cL23) |
| lsqplus_quantize_V1 | × | <=30 0.1<br><=46 0.01<br><=60 0.001<br><=70 0.0001 | 1 | -1e-9 | 69 | 90.1 | [download](https://share.weiyun.com/HRKnuJ9H) |
| lsqplu_quantize_V2 | × | <=9 0.1<br><=12 0.01 | - | - | 12 | 91.0 | [download](https://share.weiyun.com/RvrPTeEQ) |
| lsqplus_quantize_V1 | ✔ |  |  |  |  |  |  | |
| lsqplu_quantize_V2 | ✔ | <=9 0.1<br><=21 0.01<br><=33 0.001<br><=46 0.0001 | - | - | 33 | 91.46 | [download](https://share.weiyun.com/ZUTnyZJd) |
| lsqquantize_V1 | × |  |  |  |  |  | |
| lsqquantize_V2 | × |  |  |  |  |  | |
| lsqquantize_V1 | ✔ |  |  |  |  |  | |
| lsqquantize_V2 | ✔ |  |  |  |  |  | |
<br>
A represent activation, I use moving average method to initialize s and beta.<br>

LEARNED STEP SIZE QUANTIZATION<br>
LSQ+: Improving low-bit quantization through learnable offsets and better initialization<br>

### References<br>
https://github.com/666DZY666/micronet<br>
https://github.com/hustzxd/LSQuantization<br>
https://github.com/zhutmost/lsq-net<br>
https://github.com/Zhen-Dong/HAWQ<br>
https://github.com/KwangHoonAn/PACT<br>
https://github.com/Jermmy/pytorch-quantization-demo<br>
