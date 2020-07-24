using computational_graph.Layer;
using computational_graph.loss;
using FCN;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph
{
    class Program
    {
        static void Main(string[] args)
        {

             


              test2D();
            Console.ReadLine();
        }
      
        static void  test2D()
        {
            float[][][,] x = JsonConvert.DeserializeObject<float[][][,]>(getstr("D:\\x.json"));
            float[][][,] y = JsonConvert.DeserializeObject<float[][][,]>(getstr("D:\\y.json"));
            float[][][,] w = JsonConvert.DeserializeObject<float[][][,]>(getstr("D:\\w1.json"));
            float[] wb = JsonConvert.DeserializeObject<float[]>(getstr("D:\\w2.json"));

            Conv2DLayer cl = new Conv2DLayer(1, 1, 3, 1, 2, false);

            MSELoss mloss = new MSELoss();
            cl.weights = new float[w.GetLength(0)][][,] ;
            for (int a = 0; a < w.GetLength(0); a++)
            {
                cl.weights[a] = new float[w[0].GetLength(0)][,];
                for (int b = 0; b < w[a].GetLength(0); b++)
                {
                    cl.weights[a][b] = new float[0,0];
                    cl.weights[a][ b] = w[a][b];
                }
            }
            cl.basicData = wb;
            //向前传播
            dynamic temp = cl.Forward(x);

            SigmodLayer sl = new SigmodLayer();
            temp = sl.Forward(temp);

            //TanhLayer tl = new TanhLayer();
            //temp = tl.forward(temp);

            //MulLayer ml = new MulLayer();
            //temp = ml.forward(temp, y);

            float loss = mloss.Forward(temp, y);


            //向后传播
            dynamic grad = mloss.Backward();//计算误差梯度
            grad = sl.Backward(grad);
            //grad = ml.backward(grad);



            //grad = tl.backward(grad);

            dynamic grad3 = cl.backward(grad);//卷积计算在 所有计算的最后面进行

            prirt(grad3.grid);
            prirt(grad3.basic);
        }



        static void prirt(float[][][,] value)
        {
            for (var i = 0; i < value.GetLength(0); i++)
            {
                for (var j = 0; j < value[i].GetLength(0); j++)
                {
                    prirt(value[i][j]);
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }
        static void prirt(float[,] value)
        {
            for (var i = 0; i < value.GetLength(0); i++)
            {
                for (var j = 0; j < value.GetLength(1); j++)
                {
                    Console.Write(value[i, j] + ",");
                }
                Console.WriteLine();
            }
        }
        static void prirt(float[] value)
        {
            for (var i = 0; i < value.GetLength(0); i++)
            {
               
                    Console.WriteLine(value[i] + ",");
               
                
            }
        }
        static string getstr(string file)
        {
            System.IO.StreamReader sr = new System.IO.StreamReader(file);
            string str = sr.ReadToEnd();
            sr.Close();
            return str;
        }
    }
}
