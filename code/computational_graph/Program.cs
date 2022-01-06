using computational_graph.Layer;
using computational_graph.loss;
using computational_graph.Util;
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
            SSIM ssim = new SSIM();
             Matrix[] anno1 = DenseCRF.util.readpnggetMatrix("einstein.png",false);
            Matrix[] anno2 = DenseCRF.util.readpnggetMatrix("einstein2.png", false);


            //hh 误差
            float[] data = new float[] { 1,2,3,4,5 };
            float[] data2 = new float[] { 1, 2, 6, 4, 5 };
           var sim=  Sim.sim_pearson(data, data2);
            float[] ssa = new float[3];
            float ss = 0;
            for (int i = 0; i < 3; i++)
            {
                ssa[i] = ssim.ssim(anno1[i].values, anno2[i].values);
                ss += ssa[i];
                Console.WriteLine(ssa[i]);
            }
            Console.WriteLine(ss/3);

            MSELoss mSELoss = new MSELoss();
            // 1000
            var hh = mSELoss.Forward(new float[] { 2f }, new float[] { 5f }); 
            //double jiaodu= Math.Atan((37-300)/-(10 *10));
            //jiaodu = (180.0 * jiaodu / Math.PI);
            //if (jiaodu < 70) { }
            //float[] valueADList = new float[] { 0, 1, 10, 15, 24, 888 };
            //int FilterNumber = valueADList.Length;
            //for (int j = 0; j < FilterNumber - 1; j++)
            //{
            //    for (int i = 0; i < FilterNumber - 1 - j; i++)
            //    {
            //        if (valueADList[i] > valueADList[i + 1])
            //        {
            //            var temp = valueADList[i];
            //            valueADList[i] = valueADList[i + 1];
            //            valueADList[i + 1] = temp;
            //        }
            //    }
            //}

            //float sum = 0F;
            //int N = 4;//N必须为偶数
            //          //去掉N/2个最大值和最小值
            //for (int i = 0; i < FilterNumber - N ; i++)
            //{
            //    sum += valueADList[i];
            //}


            //float value=  (sum / (FilterNumber - N));




            //   test2D();
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

            dynamic grad3 = cl.Backward(grad);//卷积计算在 所有计算的最后面进行

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
