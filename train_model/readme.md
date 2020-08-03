## 1. 背景
- **增加阈值, 牺牲部分唤醒率来降低误唤醒率, PC端可行, DSCNN 效果最优, 嵌入式待补充**
- 计算唤醒率和误唤醒率的数据集(自制):
    - xrxr: 198
    - other: 163
- 除了误唤醒和唤醒率的测试之外, 其他测评数据均来自 `google speech`数据
- 网络结构均来自 [kws-streaming](https://github.com/google-research/google-research/tree/master/kws_streaming) 项目
- 以上测试方式均在 PC 端
- 上述model 均可转成 tflite, 所有 model 均未量化
- 模型推理时间未做评估


> 唤醒率( Recall ): p 个正样本中, tp 个样本被识别为正, fn 个样本被识别为负    
> recall = tp / (tp + fn)  
> 误唤醒率: n个负样本中, fp 个样本被识别为正, fn 个样本被识别为负, 其中 fp + fn = n  
>  far = fp /  (fp + fn)

## 2. 唤醒率和误唤醒率

> 左上角为最佳情况

- 未增加阈值的唤醒率和误唤醒率

![](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20200801182357.png)

- 增加阈值 0.8 和 0.9

![](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20200801182344.png)

## 3. 模型大小和准确率

![](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20200801182518.png)

## 4. 其他参数

|    Model     | Model Size | Params  |  Acc  |
| :----------: | :--------: | :-----: | :---: |
|     crnn     |   2.9 M    | 714,046 | 96.12 |
|     lstm     |   2.4 M    | 586,314 | 61.31 |
|  mobilenet   |  239.8 kb  | 36,622  | 81.57 |
|     cnn      |   2.3 M    | 607,246 | 89.43 |
|  inception   |  255.8 kb  | 12,334  | 81.69 |
| mobilenet_v2 |  247.7 kb  | 30,814  | 86.47 |
|    dscnn     |   2.1 M    | 497,412 | 93.13 |
|     dnn      |   1.8 M    | 446,860 | 90.76 |

- dataset : google speech v0.02
- labels : unknown,silence,xrxr,nihaoxr,yes,no,up,down,left,right,on,off,stop,go
- model: crnn lstm mobilenet cnn inception mobilenet_v2 dscnn dnn
- tensorflow : 2.3.0
- --window_size_ms : 40 
--window_stride_ms : 20 
--dct_num_features : 13
