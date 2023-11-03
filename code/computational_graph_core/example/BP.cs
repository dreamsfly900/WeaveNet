using computational_graph.Layer;
using computational_graph.loss;
using DenseCRF;
using FCN;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph
{
 public   class BPTest
    {
        static void Main(string[] args)
        {

            BP();


            
            Console.ReadLine();
        }
        /// <summary>
        /// BP网络测试
        /// </summary>
     public   static void BP()
        {
            String path = AppDomain.CurrentDomain.BaseDirectory;
            float[][] x = JsonConvert.DeserializeObject<float[][]>(util.getstr(Path.Combine(path, "bpx.json")));//训练数据
            float[][] y = JsonConvert.DeserializeObject<float[][]>(util.getstr(Path.Combine(path, "bpy.json")));//训练标签
            float[][] w1 = JsonConvert.DeserializeObject<float[][]>(util.getstr(Path.Combine(path, "bpw.json")));

           


            ConvLayer cl1 = new ConvLayer(13, 5, true);
            cl1.weights = w1;
            SigmodLayer sl = new SigmodLayer();
            float lr = 0.5f;
            ConvLayer cl2 = new ConvLayer(5, 1, true);
            //SigmodLayer s2 = new SigmodLayer();
            int i = 0,a=0;
            while (a < 5000)
            {
                //i = 0;
                //while (i < 100)
                //{
                //    float[][] xx2 = new float[1][];
                //    xx2[0] = new float[x[0].GetLength(0)];

                //    for (var f = 0; f < x[0].GetLength(0); f++)
                //    {

                //        xx2[0][f] = x[i][f];
                //    }
                    dynamic ff = cl1.Forward(x);
                    ff = sl.Forward(ff);
                    ff = cl2.Forward(ff);
                    // dynamic ff22 = s2.forward(ff);
                    //计算误差
                    MSELoss mloss = new MSELoss();
                    //float[][] yy2= new float[1][];
                    //yy2[0] = y[i];
                    var loss = mloss.Forward(ff, y);

                    Console.WriteLine("误差:" + loss);

                    dynamic grid = mloss.Backward();

                    //反传播w2
                    //  dynamic grid2 =s2.backward(grid);
                    dynamic w22 = cl2.backweight(grid);

                    //反传播W1
                    dynamic grid1 = cl2.backward(grid);
                    grid1 = sl.Backward(grid1);
                    dynamic w11 = cl1.backweight(grid1);


                    cl2.weights = Matrix.MatrixSub(cl2.weights, Matrix.multiply(w22.grid, lr));
                    cl2.basicData = Matrix.MatrixSub(cl2.basicData, Matrix.multiply(w22.basic, lr));

                    cl1.weights = Matrix.MatrixSub(cl1.weights, Matrix.multiply(w11.grid, lr));
                    cl1.basicData = Matrix.MatrixSub(cl1.basicData, Matrix.multiply(w11.basic, lr));
                    i++;
               // }
                a++;
            }

            //测试网络
            float[][] xx = new float[1][];
            xx[0] = new float[x[0].GetLength(0)];
            var aa = 3;
            for (var f = 0; f < x[0].GetLength(0); f++)
            {

                xx[0][f] = x[aa][f];
            }
            dynamic ff2 = cl1.Forward(xx);
            ff2 = sl.Forward(ff2);
            ff2 = cl2.Forward(ff2);

            Console.WriteLine("预测数据：");
            util.prirt(ff2);
            Console.WriteLine("期望数据：");
            util.prirt(y[aa]);
        }
     

        
    }
}
