#  WeaveNet

#### 介绍
一个使用C#编写的用于神经网络的计算图框架computational graph。带有cnn,bp,fcn,lstm,convlstm等示例。使用方法接进pytorch。

#### 软件架构
 架构完全使用c#编写，可以看到内部任何细节的实现，包含cnn,bp,fcn,lstm,convlstm等示例内容，包含示例所用的数据内容。
各项功能都在进行或者完事中，欢迎您参与此项事业，可与我联系：QQ群17375149，QQ20573886，emailL:xingyu900@live.com



 
#### 使用说明

1.  LOSS支持：MESLOSS,cross-entropy
2.  激活函数支持：ReLu，Tanh，Sigmod，Softmax
3.  数据类型支持： float[][] 与 float[][][,]，二维与四维
4.  池化支持：平均池化，最大池化
5.  其他支持：ConvLayer，Conv2DLayer，MulLayer

 部分BP代码示例

```
          //声明两个ConvLayer 和一个激活函数SigmodLayer 
          ConvLayer cl1 = new ConvLayer(13, 5, true);
          
            SigmodLayer sl = new SigmodLayer();
            float lr = 0.5f;
            ConvLayer cl2 = new ConvLayer(5, 1, true);
            
            int i = 0,a=0;
            while (a < 5000)
            {
                 
                    dynamic ff = cl1.Forward(x);
                    ff = sl.Forward(ff);
                    ff = cl2.Forward(ff);
                   
                    //计算误差
                    MSELoss mloss = new MSELoss();
                   
                    var loss = mloss.Forward(ff, y);

                    Console.WriteLine("误差:" + loss);

                    dynamic grid = mloss.Backward();

                    //反传播w2
                   
                    dynamic w22 = cl2.backweight(grid);

                    //反传播W1
                    dynamic grid1 = cl2.backward(grid);
                    grid1 = sl.Backward(grid1);
                    dynamic w11 = cl1.backweight(grid1);
                       
                   //更新参数
                    cl2.weights = Matrix.MatrixSub(cl2.weights, Matrix.multiply(w22.grid, lr));
                    cl2.basicData = Matrix.MatrixSub(cl2.basicData, Matrix.multiply(w22.basic, lr));

                    cl1.weights = Matrix.MatrixSub(cl1.weights, Matrix.multiply(w11.grid, lr));
                    cl1.basicData = Matrix.MatrixSub(cl1.basicData, Matrix.multiply(w11.basic, lr));
                    i++;
              
                a++;
            }
```

 
