using computational_graph.Layer;
using computational_graph.loss;
using DenseCRF;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
  public  class 交叉损失函数测试
    {
        static void Main(string[] args)
        {



            //ConvTranspose2DLayer ct2d = new ConvTranspose2DLayer(2,1, 3, 1, 1);
            Conv2DLayer ct2d = new Conv2DLayer(1, 1, 3, 1, 1);
            ct2d.basicData = new float[] { 0.0882f };
            ct2d.weights = new float[1][][,];
            ct2d.weights[0] = new float[1][,];
            ct2d.weights[0][0] = new float[,] { 
                { -0.0025f,  0.1788f, -0.2743f },
                { -0.2453f, -0.1284f,  0.0894f }
                ,{-0.0066f,  0.2643f, -0.0296f } };
            float[][][,] dd = new float[1][][,];
         
            float[][][,] dd2 = new float[1][][,];
            dd2[0] = new float[1][,];
            dd2[0][0] = new float[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    dd2[0][0][i, j] = 1;

            dd[0] = new float[1][,];
            dd[0][0] = new float[2, 2];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    dd[0][0][i, j] = 1;
            dynamic data= ct2d.Forward(dd2);
            util.prirt(data);
           
           // MSELoss mSELoss = new MSELoss();
           //var loss=  mSELoss.Forward(dd2,data);

           // //dynamic grid = ct2d.Backward(data);
           // //util.prirt(grid);
           // dynamic grid2 = mSELoss.Backward();
           // dynamic grid3= ct2d.Backward(grid2);
           // util.prirt(grid3);
           // dynamic weight = ct2d.backweight(grid2);
           // util.prirt(weight.grid);
            //float[] grid = new float[] { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            //Softmax softmax = new Softmax();
            //var sss = softmax.Forward(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9});
            //var ss2 = softmax.Backward(grid);


            //float[][] grid2 = new float[2][] ;
            //grid2[0] = new float[] { 11, 12, 13, 14, 15};
            //grid2[1] = new float[] { 1, 1, 1, 0,1 };
            //float[][] data = new float[2][];
            //data[0] = new float[] { 1, 2, 3, 4, 5 };
            //data[1] = new float[] {  6, 7, 8, 9 ,10};

            //Softmax softmax2 = new Softmax();
            //var ss = softmax2.Forward(data);
            //var s2 = softmax2.Backward(grid2);



            //float[,] grid3 = new float[2, 5] { { 11, 12, 13, 14, 15 } , { 1, 1, 1, 0, 1 } };

            //float[,] data3 = new float[2,5] { { 1, 2, 3, 4, 5 } , { 6, 7, 8, 9, 10 } };


            // softmax2 = new Softmax();
            //var ss22 = softmax2.Forward(data3);
            //var s22 = softmax2.Backward(grid3);

            Console.ReadLine();
        }
    }
}
