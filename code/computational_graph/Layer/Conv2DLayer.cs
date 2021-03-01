using ConsoleApp1;
using DenseCRF;
using FCN;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;  
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{
    public enum Activfunction { Null, Sigmod, Tanh, ReLU }
   
   public class Conv2DLayer
    {
        public int inChannels;   //输入图像的数目
        public int outChannels;  //输出图像的数目
        public int stride = 1;
        public int padding = 0;
        public dynamic grid;
        
        public int Kszie;
        public float[][][,] weights;
        bool full = false;
        public float[] basicData;
        public float[][] wdata;
        int inweightSize;
        bool basic;
        //Activfunction Activfunction;
        //dynamic ActaLayers;
        public Conv2DLayer(int _stride=1, int _padding=0, int kernel_size = 5, 
            int in_channels = 1, int out_channels = 6, bool initW = true,bool _full=false, int _inSize = 4,bool bias = true)
            //Activfunction _Activfunction = Activfunction.Null)
        {
            basic = bias;
            stride = _stride;
            padding = _padding;
            inChannels = in_channels;
            outChannels = out_channels;
            inweightSize = kernel_size;
            full = _full;
            //Activfunction = _Activfunction;
            init_Kszie(in_channels, out_channels, kernel_size, initW, _full, _inSize);
        }
       void init_Kszie(int innum = 1, int outnum = 6, int _weightSize = 3,bool initW = true, bool _full = false, int _inSize = 4)
        {
            inweightSize = _inSize;
            Kszie = _weightSize;
              inChannels = innum;
            outChannels = outnum;
            full = _full;
            if (initW)

                if (full)
                {
                    wdata = new float[outnum][];
                    Random rand = new Random();
                    for (int i = 0; i < outnum; i++)
                    {
                        wdata[i] = new float[inChannels * inweightSize * inweightSize];
                        for (int j = 0; j < innum* inweightSize* inweightSize; j++)
                        {
                            float randnum = (((float)rand.Next() / (float)Int32.MaxValue) - 0.5f) * 2; // 产生一个-1到1的随机数
                            wdata[i][j] = randnum * (float)Math.Sqrt(6.0f / (float)(inChannels+ outChannels));
                           // wdata[i][j] = ((float)rand.Next() / Int32.MaxValue) * 0.1f; ;
                        }
                    }
                }
                else
                {
                    weights = util.initweight(Kszie, Kszie, outnum, innum);
                }
            //y = new Matrix[outnum];
            //d = new Matrix[outnum];
            //v = new Matrix[outnum];
            basicData = new float[outnum];
            if (initW && basic)
            {
                Random rand = new Random();
                for (var b = 0; b < outnum; b++)
                    basicData[b] = ((float)rand.Next(Int32.MinValue, Int32.MaxValue) / Int32.MaxValue) * 0.1f;
            }
        }


        dynamic inputDatamatrices;
        public dynamic Forward(dynamic matrices)
        {
            //if(matrices is float[,])
            //    matrices 
            if (full)
            {
                return   forfull(matrices);
            }
            else
            {
                inputDatamatrices = matrices;
                return conv( matrices);
            }
        }
        int outSizer;
        int outSizec ;
        float[][] O5inData;
        public dynamic forfull(float[][][,] matrices)
        {
            int S4Channels = matrices.Length;
            int S4outChannels = matrices[0].Length;
            O5inData = new float[S4Channels][];
             outSizer = matrices[0][0].GetLength(0);
             outSizec = matrices[0][0].GetLength(1);
            for (int g = 0; g < (S4Channels); g++)
            {
                O5inData[g] = new float[inChannels* inweightSize* inweightSize];
                for (int i = 0; i < (S4outChannels); i++)
                    for (int r = 0; r < outSizer; r++)
                        for (int c = 0; c < outSizec; c++)
                        {
                            O5inData[g][i * outSizer * outSizec + r * outSizec + c] = matrices[g][i][r, c];
                        }
            }

            //全连接，把12*（4*4）拆成一维192长度，O5.wdata,为10种分类创建一个10*192的权重，每种权重与192 卷积，得到 10个 概率。
            //全卷积 把 12个（4*4）与 （N*N）权重，卷积后累加合成一个，如果说，输入12个MAP，输出10个，输入的12个先卷积，
            //后累加，变成一个MAP，一共10次，得到 10个 概率。
            float[][] outdata = new float[S4Channels][];
            for (int g = 0; g < (S4Channels); g++)
            {
                outdata[g]= nnffall(O5inData[g], wdata, basicData, inChannels, outChannels);
            }
            //if (Activfunction == Activfunction.Sigmod)
            //{
            //    ActaLayers = new SigmodLayer();
               
            //}
            //if (Activfunction != Activfunction.Null)
            //{
            //    outdata = ActaLayers.Forward(outdata);
            //}
            for (int g = 0; g < (S4Channels); g++)
            {
               
              
                    outdata[g] = Matrix.MatrixAdd(outdata[g], basicData);
            }
            return outdata;
        }
        float[] nnffall(float[] input, float[][] wdata, float[] bas, int inc, int outc)
        {

            float[] outadata= new float[outc];
            int i;
            for (i = 0; i < outc; i++)
            {

                outadata[i] = vecMulti(input, wdata[i], input.Length) + bas[i];
            }
            return outadata;
        }
        float vecMulti(float[] vec1, float[] vec2, int vecL)// 两向量相乘
        {
            int i;
            float m = 0;
            for (i = 0; i < vecL; i++)
                m = m + vec1[i] * vec2[i];
            return m;
        }
        public dynamic backweight(dynamic grid)
        {

            if (full)
            {
                return backfull(grid);
            }
            else
            {

                return backconv(grid);
            }
        }

        private dynamic backfull(dynamic grid)
        {
           
            int S4Channels = O5inData.Length;
            float[][] data = new float[outChannels][];
            float[] outputB = new float[outChannels];
            
            for (int g = 0; g < (S4Channels); g++)
            {
                //data = new float[outChannels][];
               
                for (int j2 = 0; j2 < outChannels; j2++)
                {
                    data[j2] = new float[inChannels * inweightSize * inweightSize]; 
                    for (int j = 0; j < inChannels * inweightSize * inweightSize; j++)
                    {
                        data[j2][j] += grid[g][j2] * O5inData[g][j];
                      
                    }
                    outputB[j2] += grid[g][j2];
                }
               
            }
            data= Matrix.divide(data, S4Channels * outChannels);
            outputB = Matrix.divide(outputB, S4Channels * outChannels);
            //sum / inputData.Length / outChannels
            var gridd = new { grid = data, basic = outputB };
            this.grid = gridd;
            return gridd;
        }

        public dynamic Backward(dynamic grid)
        {

            if (full)
            {
                return backfullY(grid);
            }
            else
            {

                return backconvY(grid);
            }
        }
        public void update(float lr = 0.1f)
        {
            if (full)
            {
                wdata = Matrix.MatrixSub(wdata, Matrix.multiply(grid.grid, lr));
                basicData = Matrix.MatrixSub(basicData, Matrix.multiply(grid.basic, lr));
            }
            else
            {

                weights = Matrix.MatrixSub(weights, Matrix.multiply(grid.grid, lr));
                basicData = Matrix.MatrixSub(basicData, Matrix.multiply(grid.basic, lr));
            }
            this.grid = null;
            
        }
       dynamic backfullY(float[][] grid)
        {
            //if (Activfunction != Activfunction.Null)
            //{
            //    grid =  ActaLayers.Backward(grid);
            //}
            int Channels = grid.Length;
            float[][][,] ddata = new float[Channels][][,];
          //  float[][] grid2 = new float[Channels][];
            for (int g = 0; g < Channels; g++) {
                float[][,] d = new float[inChannels][,];
               
                 
                for (int i = 0; i < inChannels; i++)
                {
                    for (int r = 0; r < outSizer; r++)
                        for (int c = 0; c < outSizec; c++)
                            for (int j = 0; j < outChannels; j++)
                            {
                                int wInt = i * outSizec * outSizer + r * outSizec + c;
                                if (d[i] == null)
                                    d[i] = new float[outSizer, outSizec];
                                d[i][r, c] = d[i][r, c] + grid[g][j] * wdata[j][wInt];
                                 
                            }
                }
                
                ddata[g] = d;
            }
            return ddata;
        }
        dynamic backconv(float[][][,] grid)
        {
            int i, j;

           // if (grid is float[][][,])
            {
                float[][][,] inputData = grid;

                dynamic outputDataall = null;
                //Parallel.For(0, inputData.Length, cc =>
                 for (var cc = 0; cc < inputData.Length; cc++)
                {
                    float[][][,] outputData = new float[outChannels][][,];
                    // if (!Matrix.CUDA)
                    {
                        DateTime stat = DateTime.Now;
                        //outputData = new float[outChannels][][,];
                        for (i = 0; i < (outChannels); i++)
                        {
                            outputData[i] = new float[inChannels][,];
                            for (j = 0; j < (inChannels); j++)
                            {
                                float[,] v1 = null;

                                //  float[,] tt=  Matrix.rot180(grid[cc][i]);
                                v1 = Matrix.convnValid(inputDatamatrices[cc][j], grid[cc][i], stride, padding);
                                // v1 = Matrix.convolution(inputDatamatrices[cc][j], grid[cc][i], stride, padding);
                                if (outputData[i][j] != null)
                                    outputData[i][j] = Matrix.MatrixAdd(outputData[i][j], v1).values;
                                else
                                    outputData[i][j] = v1;


                                // Matrix.MatrixAdd(outputData[i][j], basicData[i]);

                            }

                        }
                        DateTime end = DateTime.Now;
                        Console.WriteLine($"backconv计算时间：{(end - stat).TotalMilliseconds}");



                        // outputData = Matrix.divide(outputData, outChannels);

                    }
                    //else
                    //{


                    //}
                    if (outputDataall == null)
                        outputDataall = outputData;
                    else
                    {
                        outputDataall = Matrix.MatrixAdd(outputDataall, outputData);
                    }

                }
              //   );
               // outputDataall = Matrix.divide(outputDataall, inputData.Length* outChannels);
                float[] outputB = new float[outChannels];
                if (basic)
                {
                    for (var s = 0; s < outChannels; s++)
                    {


                   
                            float sum = 0f;
                            for (var cc = 0; cc < inputData.Length; cc++)
                            {
                                sum += Matrix.sum(inputData[cc][s]);


                            }
                        // outputB[s] = sum / inputData.Length / outChannels;

                        outputB[s] = sum / inputData.Length;
                        //outputB[s] = sum;
                    }
                }
             

               var  gridd = new { grid = outputDataall, basic = outputB };
                this.grid = gridd;
                //if (this.grid == null)
                   
                //else
                //{
                //    this.grid = new { grid = Matrix.MatrixAdd(this.grid.grid, gridd.grid), 
                //        basic = Matrix.MatrixAdd(this.grid.basic, gridd.basic) };
                //}
                return gridd;
            }
        }
      //  dynamic gridk;
        dynamic backconvY(float[][][,] grid)
        {
            int i;
            //if (Activfunction != Activfunction.Null)
            //    grid = ActaLayers.Forward(grid);
            // if (grid is float[][][,])
            {
                float[][][,] inputData = grid;
                float[][][,] outputData = new float[inputData.Length][][,];
                dynamic outputDataall = null;

                for (var cc = 0; cc < inputData.Length; cc++)
              //  Parallel.For(0, inputData.Length, cc =>
                {
                    outputData[cc] = new float[inChannels][,];
                    for (i = 0; i < (inChannels); i++)
                    {
                        for (int j = 0; j < (outChannels); j++)

                        // Parallel.For(0, outChannels, j =>

                        {
                            float[,] v1 = null;
                            float[,] weight = Matrix.rot180(weights[j][i]);

                            v1 = Matrix.convnFull(inputData[cc][j], weight, stride, padding);


                            if (outputData[cc][i] != null)
                                outputData[cc][i] = Matrix.MatrixAdd(outputData[cc][i], v1).values;
                            else
                                outputData[cc][i] = v1;


                            // Matrix.MatrixAdd(outputData[i][j], basicData[i]);

                        }

                        //);



                    }
                    //  outputData = Matrix.divide(outputData, outChannels);
                    if (outputDataall == null)
                        outputDataall = outputData;
                    else
                    {
                        outputDataall = Matrix.MatrixAdd(outputDataall, outputData);
                    }


                }
                //);
                if(inputData.Length>1)
                outputDataall = Matrix.divide(outputDataall, inputData.Length);
               
                return outputDataall;
            }
        }
        CudaDeviceVariable<float> d_A;
        CudaDeviceVariable<float> d_B;
        CudaDeviceVariable<float> d_C;
        float[][][,] conv( float[][][,] inputData)
        {
            float[][][,] outputData = new float[inputData.Length][][,];
            int i;


            if (Matrix.CUDA)
            {
                int k = weights[0][0].GetLength(0);

                int w = inputData[0][0].GetLength(0);
                int h = inputData[0][0].GetLength(1);
                var row = ((w - k) + 2 * padding) / stride + 1;
                var col = ((h - k) + 2 * padding) / stride + 1;
                d_C = new float[((outChannels * inChannels) * row * col)];
            }
            //if (Activfunction == Activfunction.Sigmod)
            //{
            //    ActaLayers = new SigmodLayer();

            //}
            for (var cc = 0; cc < inputData.Length; cc++)
            // Parallel.For(0, inputData.Length, cc =>
            {
             
                if (!Matrix.CUDA)
                {
                    outputData[cc] = new float[outChannels][,];
                    DateTime stat = DateTime.Now;
                    for (i = 0; i < (outChannels); i++)
                    {

                        float[,] v1 = null;
                        for (int j = 0; j < (inChannels); j++)
                        // Parallel.For(0, inChannels, j =>
                        {

                            float[,] temp;

                            temp = Matrix.convnValid(inputData[cc][j], weights[i][j], stride, padding);
                            // temp = Matrix.convolution(inputData[cc][j], weights[i][j], stride, padding);
                            if (v1 == null)
                                v1 = new float[temp.GetLength(0), temp.GetLength(1)];
                            v1 = Matrix.MatrixAdd(v1, temp).values;
                            //temp = inputData[j].convolution(C1.weights[j, i].values, C1.stride, C1.padding);//向前传播 
                            outputData[cc][i] = v1;
                            // });
                        }
                        //  outputData[cc][i] = Matrix.MatrixAdd(v1, basicData[i]).values;

                    }
                    DateTime end = DateTime.Now;
                   // Console.WriteLine($"计算时间：{(end - stat).TotalMilliseconds}");
                }
                else
                {
                    DateTime stat = DateTime.Now;
                    float[] h_A = Matrix.float3DTofloat1D(inputData[cc]);
                    d_A = h_A;
                    float[] h_B = Matrix.float4DTofloat1D(weights);
                    DateTime end = DateTime.Now;
                 //   Console.WriteLine($"复制数据计算时间：{(end - stat).TotalMilliseconds}");
                    d_B = h_B;
                    int k = weights[0][0].GetLength(0);

                    int w = inputData[cc][0].GetLength(0);
                    int h = inputData[cc][0].GetLength(1);
                    var row = ((w - k) + 2 * padding) / stride + 1;
                    var col = ((h - k) + 2 * padding) / stride + 1;

                    //Ho=(H−F+2×P)/S+1

                    int O_TILE_WIDTH = 16;

                    if (outChannels < O_TILE_WIDTH && inChannels < O_TILE_WIDTH)
                        O_TILE_WIDTH = Math.Max(outChannels, inChannels);

                    int BLOCK_WIDTH = O_TILE_WIDTH;


                    Matrix.CUDA3dConvKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(BLOCK_WIDTH, BLOCK_WIDTH);

                    Matrix.CUDA3dConvKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((outChannels - 1) / O_TILE_WIDTH + 1, (inChannels - 1) / O_TILE_WIDTH + 1);
                    //   Matrix.CUDA3dConvKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(5,1);
                    Matrix.CUDA3dConvKernel.SetConstantVariable("O_TILE_WIDTH", O_TILE_WIDTH);
                    stat = DateTime.Now;
                    //   Matrix.CUDA3dConvKernel.DynamicSharedMemory = (uint)((outChannels * inChannels) * row * col);
                    //  Matrix.CUDA3dConvKernel.MaxDynamicSharedSizeBytes = (int)Matrix.CUDA3dConvKernel.DynamicSharedMemory;
                    //Class1.convolution_3D_shared(h_A, h_B, new float[(outChannels * inChannels) * row * col], k, w, h, row, col, stride, padding, outChannels, inChannels);
                    Matrix.CUDA3dConvKernel.Run(d_A.DevicePointer, d_B.DevicePointer, d_C.DevicePointer,
                        k, w, h, row, col, stride, padding, outChannels, inChannels);
                    end = DateTime.Now;
                   // Console.WriteLine($"CUDA计算conv时间：{(end - stat).TotalMilliseconds}");
                    float[] gg = d_C;

                    float[][,] temp = new float[outChannels][,];
                    stat = DateTime.Now;
                    // Parallel.For(0, inputData.Length, inlen =>
                    for (int inlen = 0; inlen < (outChannels); inlen++)
                    {
                        int len = (row * col) * outChannels;
                        int index = inlen * (row * col) * outChannels;
                        temp[inlen] = new float[row, col];
                        for (int olen = 0; olen < (inChannels); olen++)
                        {
                            for (int s = 0; s < (row * col); s++)
                            {
                                int r = s / row;
                                int c = (s) % row;

                                temp[inlen][r, c] += gg[(inlen * inChannels * row * col) + (olen * row * col) + (r * row) + c];
                            }

                        }


                    }//);
                    end = DateTime.Now;
                  //  Console.WriteLine($"CPU换算计算时间：{(end - stat).TotalMilliseconds}");
                    outputData[cc] = temp;
                    d_A.Dispose();
                    d_B.Dispose();
                    d_C.Dispose();
                }

            }
           
            // );
            //if (Activfunction != Activfunction.Null)
            //    outputData = ActaLayers.Forward(outputData);
            if (basic)
            {
                for (var cc = 0; cc < inputData.Length; cc++)
                {
                    for (i = 0; i < (outChannels); i++)
                    {
                        outputData[cc][i] = Matrix.MatrixAdd(outputData[cc][i], basicData[i]).values;
                    }
                }
            }
           
              return outputData;
        }
     

    }
}
