# LSQ and LSQ+<br>
LSQ+ net or LSQplus net and LSQ net <br>

## commit log<br>
`
2023-01-08
`
Dorefa and Pact, [https://github.com/ZouJiu1/Dorefa_Pact](https://github.com/ZouJiu1/Dorefa_Pact)<br>
--------------------------------------------------------------------------------------------------------------<br>
add torch.nn.Parameter .data, retrain models 18-01-2022<br>

I'm not the author, I just complish an unofficial implementation of LSQ+ or LSQplus and LSQ，the origin paper you can find LSQ+ here [arxiv.org/abs/2004.09576](https://arxiv.org/abs/2004.09576) and LSQ here [arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153).<br>

pytorch==1.8.1<br>

You should train 32-bit float model firstly, then you can finetune a low bit-width quantization QAT model by loading the trained 32-bit float model<br>

Dataset used for training is CIFAR10 and model used is Resnet18 revised<br>

## Version introduction
lsqplus_quantize_V1.py: initialize s、beta of activation quantization according to LSQ+ [LSQ+: Improving low-bit quantization through learnable offsets and better initialization](https://arxiv.org/abs/2004.09576)<br><br>
lsqplus_quantize_V2.py: initialize s、beta of activation quantization according to min max values<br><br>
lsqquantize_V1.py：initialize s of activation quantization according to LSQ [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)<br><br>
lsqquantize_V2.py: initialize s of activation quantization = 1<br><br>
lsqplus_quantize_V2.py has the best result when use cifar10 dataset<br>

## The Train Results 
### For the below table all set a_bit=8, w_bit=8
| version | weight per_channel | learning rate | A s initial | A beta initial | best epoch | Accuracy | models
| ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Float 32bit | - | <=66 0.1<br><=86 0.01<br><=99 0.001<br><=112 0.0001 | - | - | 112 | 92.6 | [https://www.aliyundrive.com/s/6B2AZ45fFjx](https://www.aliyundrive.com/s/6B2AZ45fFjx) |
| lsqplus_quantize_V1 | × | <=31 0.1<br><=61 0.01<br><=81 0.001<br><112 0.0001 | 1 | -1e-9 | 90 | 90.3 | [https://www.aliyundrive.com/s/FNZRhoTe8uW](https://www.aliyundrive.com/s/FNZRhoTe8uW) |
| lsqplus_quantize_V2 | × | as before | - | - | 87 | 92.8 | [https://www.aliyundrive.com/s/WDH3ZnEa7vy](https://www.aliyundrive.com/s/WDH3ZnEa7vy) |
| lsqplus_quantize_V1 | ✔ | as before | - | - | 96 | 91.19  | [https://www.aliyundrive.com/s/JATsi4vdurp](https://www.aliyundrive.com/s/JATsi4vdurp) |
| lsqplus_quantize_V2 | ✔ | as before | - | - | 69 | 92.8 | [https://www.aliyundrive.com/s/LRWHaBLQGWc](https://www.aliyundrive.com/s/LRWHaBLQGWc) |
| lsqquantize_V1 | × | as before | - | - | 102 | 91.89 | [https://www.aliyundrive.com/s/nR1KZZRuB23](https://www.aliyundrive.com/s/nR1KZZRuB23) |
| lsqquantize_V2 | × | as before | - | - | 69 | 91.82 | [https://www.aliyundrive.com/s/7fjmViqUvh4](https://www.aliyundrive.com/s/7fjmViqUvh4) |
| lsqquantize_V1 | ✔ | as before | - | - | 108 | 91.29 | [https://www.aliyundrive.com/s/](https://www.aliyundrive.com/s/PX84qGorVxY) |
| lsqquantize_V2 | ✔ | as before | - | - | 72 | 91.72 | [https://www.aliyundrive.com/s/7nGvMVZcKp7](https://www.aliyundrive.com/s/7nGvMVZcKp7) |
<br>
all

[https://www.aliyundrive.com/s/hng9XsvhYru](https://www.aliyundrive.com/s/hng9XsvhYru)  

<br>
A represent activation, I use moving average method to initialize s and beta.<br><br>

LEARNED STEP SIZE QUANTIZATION<br>
LSQ+: Improving low-bit quantization through learnable offsets and better initialization<br>

### References<br>
https://github.com/666DZY666/micronet<br>
https://github.com/hustzxd/LSQuantization<br>
https://github.com/zhutmost/lsq-net<br>
https://github.com/Zhen-Dong/HAWQ<br>
https://github.com/KwangHoonAn/PACT<br>
https://github.com/Jermmy/pytorch-quantization-demo<br>
