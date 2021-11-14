# LSQ and LSQ+<br>
LSQ+ or LSQplus and LSQ <br>

I'm not the author, I just complish an unofficial implementation of LSQ+ or LSQplus and LSQ，the origin paper you can find LSQ+ here [arxiv.org/abs/2004.09576](https://arxiv.org/abs/2004.09576) and LSQ here [arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153).<br>

pytorch==1.8.0<br>

You should train 32-bit float model firstly, then you can finetune a low bit-width quantization QAT model by loading the trained 32-bit float model<br>

Dataset is CIFAR10 and model used is Resnets<br>

LEARNED STEP SIZE QUANTIZATION<br>
LSQ+: Improving low-bit quantization through learnable offsets and better initialization<br>

### References<br>
https://github.com/666DZY666/micronet<br>
https://github.com/hustzxd/LSQuantization<br>
https://github.com/zhutmost/lsq-net<br>
https://github.com/Zhen-Dong/HAWQ<br>
https://github.com/KwangHoonAn/PACT<br>
https://github.com/Jermmy/pytorch-quantization-demo<br>
