using computational_graph.Layer;
using DenseCRF;
using FCN;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
    class GPUtest
    {
        static void Main(string[] args)
        {

            //   int w = 5;
            //   int h = 5;
            //   int k = 3;

            //   float[,] aa = new float[w, h];
            float[][][,] aa = Matrix.zroe2D(1, 2, 5, 5);

            aa = util.initweight(5, 5, 1, 2);
            //  // aa = Matrix.init2Ddata(aa, 1);
            //   aa = Matrix.randinit(aa, 1);
            //   float[,] yy = new float[k, k];
            //   yy = Matrix.randinit(yy,1);
            //   DateTime stat;
            //   DateTime end;
            //   float[,] outconv;
            //  // while (true)
            //   {
            //        stat = DateTime.Now;
            //       Matrix.CUDA = false;
            //       outconv = Matrix.convolution(aa, yy, 1, 0);
            //        end = DateTime.Now;
            //       Console.WriteLine($"计算时间：{(end - stat).TotalMilliseconds}");
            //   }
            Matrix.CUDA = true; 
            ////   while (true)
            //   {
            //       stat = DateTime.Now;
            //       float[,] kk = Matrix.convolution(aa, yy, 1, 0);
            //       end = DateTime.Now;
            //       bool bb = check(kk, outconv);
            //       if(bb)
            //       Console.WriteLine($"计算时间：{(end - stat).TotalMilliseconds}");
            //       else Console.WriteLine($"错误：{(end - stat).TotalMilliseconds}");
            //   }
            Conv2DLayer conv2D = new Conv2DLayer(1, 0, 3, 2, 2);
            dynamic bb= conv2D.Forward(aa);
              
            Matrix.CUDA = false;
            dynamic bb2 = conv2D.Forward(aa);
         }
        public bool check(float[] h_C, float[] h_B)
        {
            float sum = 0;
            for (int i = 0; i < h_B.Length; ++i)
                sum += h_B[i];
            for (int i = 0; i < h_C.Length; ++i)
                if (h_C[i] != sum)
                    return false;

            return true;

        }
        public static bool check(float[,] h_C, float[,] h_B)
        {
            float sum = 0;
            for (int i = 0; i < h_B.GetLength(0); ++i)
                for (int j = 0; j < h_B.GetLength(1); ++j)
                {
                    if (h_C[i, j] != h_B[i, j])
                        return false;
                }

            return true;

        }
    }
}
