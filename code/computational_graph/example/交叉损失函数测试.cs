using computational_graph.Layer;
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
            float[] grid = new float[] { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            Softmax softmax = new Softmax();
            var sss = softmax.Forward(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9});
            var ss2 = softmax.Backward(grid);


            float[][] grid2 = new float[2][] ;
            grid2[0] = new float[] { 11, 12, 13, 14, 15};
            grid2[1] = new float[] { 1, 1, 1, 0,1 };
            float[][] data = new float[2][];
            data[0] = new float[] { 1, 2, 3, 4, 5 };
            data[1] = new float[] {  6, 7, 8, 9 ,10};
           
            Softmax softmax2 = new Softmax();
            var ss = softmax2.Forward(data);
            var s2 = softmax2.Backward(grid2);



            float[,] grid3 = new float[2, 5] { { 11, 12, 13, 14, 15 } , { 1, 1, 1, 0, 1 } };
            
            float[,] data3 = new float[2,5] { { 1, 2, 3, 4, 5 } , { 6, 7, 8, 9, 10 } };
           

             softmax2 = new Softmax();
            var ss22 = softmax2.Forward(data3);
            var s22 = softmax2.Backward(grid3);
        }
    }
}
