# LSQ and LSQ+
LSQ+ or LSQplus and LSQ 

I'm not the author, I just complish an unofficial implementation of LSQ+ or LSQplus and LSQ，the origin paper you can find LSQ+ here [arxiv.org/abs/2004.09576](https://arxiv.org/abs/2004.09576) and LSQ here [arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153).<br>


You should train 32-bit float model firstly, then you can finetune a low bit-width quantization QAT model by loading the trained 32-bit float model

Dataset is CIFAR10 and model used is Resnets<br>


LSQ+: Improving low-bit quantization through learnable offsets and better initialization<br>

### References
https://github.com/666DZY666/micronet
https://github.com/hustzxd/LSQuantization
https://github.com/zhutmost/lsq-net
https://github.com/Zhen-Dong/HAWQ
https://github.com/KwangHoonAn/PACT
https://github.com/Jermmy/pytorch-quantization-demo
