using computational_graph.Layer;
using computational_graph.loss;
using DenseCRF;
using FCN;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
    public class ConvLSTMtest2
    {
        static void Main(string[] args)
        {
            tian();


        }
        static void tian()
        {
            string[] files = System.IO.Directory.GetFiles("test");
            files = files.OrderBy(p => p).ToArray();
            float[][][,] datax = new float[1][][,];
          
            float[][][,] datay = new float[1][][,];
            float[][][,] h_prev=null ;
            float[][][,] c_prev = null;
            datax[0] = new float[10][,]; 
            datay[0] = new float[10][,];
            MSELoss mloss = new MSELoss();
            ConvLSTMCell2 convLSTM = new ConvLSTMCell2(3);
            
        //  convLSTM.load("radar.bin");
        // convLSTM.load("convLSTM.bin");
        lb123:
            for (int r = 0; r < files.Length - 11; r++)
            {
                
                var loss = 0.0f;
               
                for (int t = 0; t < 10; t++)
                {
                    string file = files[r + t];
                    float[,] anno1 = DenseCRF.util.readRADARMatrix(file);
                   // if(r==0)
                    anno1 = ImgUtil.BilinearInterp(anno1, 128, 128);
                  //  
                    datax[0][t] = THEmedian( anno1);
                  //  ImgUtil.savefile2(datax[0][t], "D:/testpng/a" + t + ".png");
                    if ((r + t + 10) >= files.Length)
                    { goto lb123; }
                    string file2 = files[r + t + 10];
                    datay[0][t] = DenseCRF.util.readRADARMatrix(file2);
                    datay[0][t] = THEmedian(datay[0][t]);
                    datay[0][t] = ImgUtil.BilinearInterp(datay[0][t], 128, 128);
                    

                }
                h_prev = zroe(datax);
                c_prev = zroe(datax);
               
                while (true)
                {
                  
                    var star = DateTime.Now;
                    var next = convLSTM.Forward(datax);
                    //h_prev = next;
                    //c_prev = cp;
                    for (int ss = 0; ss < 10; ss++)
                        ImgUtil.savefile2(next[0][ss], "D:/testpng/r" + (r * ss) + ss + ".png");
                    loss = mloss.Forward(next, datay);
                    var gird = mloss.Backward();
                    convLSTM.Backward(gird);
                    convLSTM.update(0.1f);
                    var end = DateTime.Now;
                    Console.WriteLine((end - star).TotalMilliseconds);
                    Console.WriteLine("lost:" + loss);
                }
                convLSTM.save("convLSTM.bin");


            }
        }
        static dynamic zroe(float[][][,] x)
        {
            float[][][,] h_prev = new float[x.Length][][,];
            for (int i = 0; i < x.Length; i++)
            {
                h_prev[i] = new float[x[i].Length][,];

                for (int j = 0; j < x[i].Length; j++)
                {
                    h_prev[i][j] = new float[x[i][j].GetLength(0), x[i][j].GetLength(1)];

                }
            }
            return h_prev;
        }
        static  float[,] THEmedian(float[,] data)
        {
            int x, y;
            float[] p = new float[9]; //最小处理窗口3*3
            float s;
            //byte[] lpTemp=new BYTE[nByteWidth*nHeight];
            int i, j;
            var row = data.GetLength(0);
            var clos = data.GetLength(1);
            //--!!!!!!!!!!!!!!下面开始窗口为3×3中值滤波!!!!!!!!!!!!!!!!
            for (y = 1; y < clos - 1; y++) //--第一行和最后一行无法取窗口
            {
                for (x = 1; x < row - 1; x++)
                {
                  //  if (data[x, y] <= 0)
                    {
                        //取9个点的值
                        p[0] = data[x - 1, y - 1];
                        p[1] = data[x, y - 1];
                        p[2] = data[x + 1, y - 1];
                        p[3] = data[x - 1, y];
                        p[4] = data[x, y];
                        p[5] = data[x + 1, y];
                        p[6] = data[x - 1, y + 1];
                        p[7] = data[x, y + 1];
                        p[8] = data[x + 1, y + 1];
                        //计算中值
                        for (j = 0; j < 5; j++)
                        {
                            for (i = j + 1; i < 9; i++)
                            {
                                if (p[j] > p[i])
                                {
                                    s = p[j];
                                    p[j] = p[i];
                                    p[i] = s;
                                }
                            }
                        }
                        ////      if (bmpobj.GetPixel(x, y).R < dgGrayValue)
                        ////  bmpobj.SetPixel(x, y, Color.FromArgb(p[4], p[4], p[4]));    //给有效值付中值
                        data[x, y] = p[4];
                    }
                    //  data[x, y] = (p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7] + p[8]) / 9;
                }
            }
            return data;
        }


      
    }
  
    
}
