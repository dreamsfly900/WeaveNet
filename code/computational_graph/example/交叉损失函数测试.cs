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



            ConvTranspose2DLayer ct2d = new ConvTranspose2DLayer(2,1, 3, 1, 1);
            Conv2DLayer con2d = new Conv2DLayer(1, 0, 2, 1, 1);
            con2d.basicData = new float[] { -0.1029f };
            con2d.weights = new float[1][][,];
            con2d.weights[0] = new float[1][,];
            con2d.weights[0][0] = new float[,] { { -0.3390f,-0.2177f },{ 0.1816f,0.4152f} };
            // Conv2DLayer ct2d = new Conv2DLayer(2, 1, 3, 1, 1);
            ct2d.basicData = new float[] { 0.2f };
            ct2d.weights = new float[1][][,];
            ct2d.weights[0] = new float[1][,];
            ct2d.weights[0][0] = new float[,] { 
                { -0.0296f,  0.0882f, -0.1007f },
                { -0.0655f, -0.3184f, -0.2208f }
                ,{-0.1374f,  0.0123f,  0.1318f} };
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
            float[][][,] dd3 = new float[1][][,];
            dd3[0] = new float[1][,];
            dd3[0][0] = new float[,] { 
                { 1.5410f, -0.2934f },
                { -2.1788f, 0.5684f } };


            dynamic data = con2d.Forward(dd);
             data = ct2d.Forward(data);
            util.prirt(data);

            MSELoss mSELoss = new MSELoss();
            var loss = mSELoss.Forward(data, dd2);//左边X,右边Y
            dynamic grid2 = mSELoss.Backward();
            grid2 = ct2d.Backward(grid2);
            // dynamic grid = ct2d.Backward(grid2);
            //  dynamic grid = mulLayer.backward(grid2);
            dynamic weight = con2d .backweight(grid2);
            util.prirt(weight.grid);
           // util.prirt(grid);
            // dynamic grid2 = mSELoss.Backward();
            // dynamic grid3= ct2d.Backward(grid2);
            // util.prirt(grid3);
        
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
