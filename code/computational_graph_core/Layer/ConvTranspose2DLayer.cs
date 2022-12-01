using DenseCRF;
using FCN;
using System;
using System.Collections.Generic;
using System.Linq;  
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{
   
   public class ConvTranspose2DLayer
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
        //Activfunction Activfunction;
        //dynamic ActaLayers;
        bool basic;
        public ConvTranspose2DLayer(int _stride=1, int _padding=0, int weightswidth = 5, 
            int innum = 1, int outnum = 6, bool initW = true, int _inSize = 4, bool bias = true)
            //Activfunction _Activfunction = Activfunction.Null)
        {
            basic = bias;
            stride = _stride;
            padding = _padding;
            inChannels = innum;
            outChannels = outnum;
            inweightSize = weightswidth;
            full = false;
            //Activfunction = _Activfunction;
            init_Kszie(innum, outnum, weightswidth, initW, full, _inSize);
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
           
                inputDatamatrices = matrices;
                return conv( matrices);
            
        }
        int outSizer;
        int outSizec ;
        float[][] O5inData;
       
        public dynamic backweight(dynamic grid)
        {

            

                return backconv(grid);
            
        }

    
        public dynamic Backward(dynamic grid)
        {

           

                return backconvY(grid);
            
        }
   
        dynamic backconv(float[][][,] grid)
        {
            int i, j;

           // if (grid is float[][][,])
            {
                float[][][,] inputData = grid;

                dynamic outputDataall = null;
                Parallel.For(0, inputData.Length, cc =>
                //  for (var cc = 0; cc < inputData.Length; cc++)
                {
                    float[][][,] outputData = new float[outChannels][][,];
                    for (i = 0; i < (outChannels); i++)
                    {
                        outputData[i] = new float[inChannels][,];
                        for (j = 0; j < (inChannels); j++)
                        {
                            float[,] v1 = null;

                            //  float[,] tt=  Matrix.rot180(grid[cc][i]);
                         
                            v1 = Matrix.Transposeconvolution(inputDatamatrices[cc][j], grid[cc][i], stride, padding, Kszie);
                            v1 = Matrix.rot180(v1);
                            if (outputData[i][j] != null)
                                outputData[i][j] = Matrix.MatrixAdd(outputData[i][j], v1).values;
                            else
                                outputData[i][j] = v1;


                            // Matrix.MatrixAdd(outputData[i][j], basicData[i]);

                        }

                    }

                    // outputData = Matrix.divide(outputData, outChannels);
                    if (outputDataall == null)
                        outputDataall = outputData;
                    else
                    {
                        outputDataall = Matrix.MatrixAdd(outputDataall, outputData);
                    }


                }
                 );
                outputDataall = Matrix.divide(outputDataall, inputData.Length* outChannels);
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
                        outputB[s] = sum / inputData.Length / outChannels;
                    }
                    //outputB[s] = sum;
                }

               var gridd = new { grid = outputDataall, basic = outputB };
                gridK = gridd;
                return gridd;
            }
        }
        public void update(float lr = 0.1f)
        {
            

                weights = Matrix.MatrixSub(weights, Matrix.multiply(gridK.grid, lr));
                basicData = Matrix.MatrixSub(basicData, Matrix.multiply(gridK.basic, lr));
            


        }
        dynamic gridK;
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

                // for (var cc = 0; cc < inputData.Length; cc++)
                Parallel.For(0, inputData.Length, cc =>
                {
                    outputData[cc] = new float[inChannels][,];
                    for (i = 0; i < (inChannels); i++)
                    {
                        for (int j = 0; j < (outChannels); j++)

                        // Parallel.For(0, outChannels, j =>

                        {
                            float[,] v1 = null;
                            float[,] weight = Matrix.rot180(weights[j][i]);
                          //  float[,] weight =(weights[j][i]);
                            v1 = Matrix.convolution(inputData[cc][j], weight, stride, padding);



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


                });
              //  if(inputData.Length>1)
                outputDataall = Matrix.divide(outputDataall, inputData.Length* outChannels);
               
                return outputDataall;
            }
        }
        float[][][,] conv( float[][][,] inputData)
        {
            float[][][,] outputData = new float[inputData.Length][][,];
            int i;


            //if (Activfunction == Activfunction.Sigmod)
            //{
            //    ActaLayers = new SigmodLayer();

            //}
            //   for (var cc = 0; cc < inputData.Length; cc++)
            Parallel.For(0, inputData.Length, cc =>
            {
                outputData[cc] = new float[outChannels][,];

                for (i = 0; i < (outChannels); i++)
                {

                    float[,] v1 = null;
                    for (int j = 0; j < (inChannels); j++)
                    // Parallel.For(0, inChannels, j =>
                    {
                        //(Height - 1) * Stride - 2 * padding + Size(Height−1)∗Stride−2∗padding + Size
                        float[,] temp;
                          temp= Matrix.rot180(weights[i][j]);
                        temp = Matrix.Transposeconvolution (inputData[cc][j], temp, stride, padding, Kszie);
                        if (v1 == null)
                            v1 = new float[temp.GetLength(0), temp.GetLength(1)];
                        v1 = Matrix.MatrixAdd(v1, temp).values;
                        //temp = inputData[j].convolution(C1.weights[j, i].values, C1.stride, C1.padding);//向前传播 
                        outputData[cc][i] = v1;
                        // });
                    }
                    //  outputData[cc][i] = Matrix.MatrixAdd(v1, basicData[i]).values;

                }

            }
            );
            //if (Activfunction != Activfunction.Null)
            //    outputData = ActaLayers.Forward(outputData);
            if (basic) 
            for (var cc = 0; cc < inputData.Length; cc++)
            {
                for (i = 0; i < (outChannels); i++)
                {
                    outputData[cc][i] = Matrix.MatrixAdd(outputData[cc][i], basicData[i]).values;
                }
            }
              return outputData;
        }
     

    }
}
