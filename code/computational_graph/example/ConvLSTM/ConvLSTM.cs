using computational_graph.Layer;
using computational_graph.loss;
using DenseCRF;
using FCN;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example.ConvLSTM
{
    public class ConvLSTMtest
    {
  
        static void Main(string[] args)
        {
           ///train();
            ceshi();
           //var ww=Gates.backweight(gird);

        }
        static void train()
        {
            string[] files = System.IO.Directory.GetFiles(@"D:\testpng\res");
            files = files.OrderBy(p => p).ToArray();
            List<float[,]> list = new List<float[,]>();
            for (int r = 0; r < files.Length; r++)
            {
                String file = files[r];
                float[,] anno1 = DenseCRF.util.readRADARMatrix(file);
                //anno1 = ImgUtil.BilinearInterp(anno1, 128, 128);
                list.Add(anno1);
            }
            ConvLSTM convLSTM = new ConvLSTM();
            convLSTM.load("ceshi.bin");
            //while (true)
            {
                for (int r = 0; r < list.Count - 10; r++)
                {
                    List<float[][][,]> listx = new List<float[][][,]>();
                    List<float[][][,]> listy = new List<float[][][,]>();
                    for (int s = 0; s < 10; s++)
                    {
                        float[][][,] x = new float[1][][,];
                        x[0] = new float[1][,];
                        x[0][0] = list[r + s];
                        listx.Add(x);
                        float[][][,] y = new float[1][][,];
                        y[0] = new float[1][,];
                        y[0][0] = list[r + s + 1];
                        listy.Add(y);
                    }
                    convLSTM.train(listx, listy);
                }
                convLSTM.save("ceshi.bin");
            }
        }
        static dynamic zroe(float[][][,] x,int len)
        {
            float[][][,] h_prev = new float[x.Length][][,];
            for (int i = 0; i < x.Length; i++)
            {
                h_prev[i] = new float[len][,];

                for (int j = 0; j < len; j++)
                {
                    h_prev[i][j] = new float[x[0][0].GetLength(0), x[0][0].GetLength(1)];
                    
                }
            }
            return h_prev;
        }
        static  dynamic ones(float[][][,] x)
        {
            float[][][,] h_prev = new float[x.Length][][,];
            for (int i = 0; i < x.Length; i++)
            {
                h_prev[i] = new float[x[i].Length][,];

                for (int j = 0; j < x[i].Length; j++)
                {
                    h_prev[i][j] = new float[x[i][j].GetLength(0), x[i][j].GetLength(1)];
                    for (int x1 = 0; x1 < x[i][j].GetLength(0); x1++)
                        for (int y1 = 0; y1 < x[i][j].GetLength(1); y1++)
                        {
                            h_prev[i][j][x1, y1] = 1;
                        }
                }
            }
            return h_prev;
        }

       static void ceshi()
        {
            string[] files = System.IO.Directory.GetFiles(@"D:\testpng\res");
            files = files.OrderBy(p => p).ToArray();
            List<float[,]> list = new List<float[,]>();
            var r = 0;
            for ( r = 0; r < files.Length; r++)
            {
                String file = files[r];
                float[,] anno1 = DenseCRF.util.readRADARMatrix(file);
                DenseCRF.ImgUtil.savefile(anno1, @"D:\testpng\A" + r + ".png");
                //anno1 = ImgUtil.BilinearInterp(anno1, 128, 128);
                list.Add(anno1);
            }
            ConvLSTM convLSTM = new ConvLSTM();
            convLSTM.load("ceshi.bin");
           
            List<float[][][,]> listx = new List<float[][][,]>();
            
          //  for (int r = 0; r < list.Count - 10; r++)
            {
              
                for (int s = 5; s < 15; s++)
                {
                    float[][][,] x = new float[1][][,];
                    x[0] = new float[1][,];
                    x[0][0] = list[ s];
                    listx.Add(x);
                    
                }
                convLSTM.test(listx);
            }

        }
    }
    public class ConvLSTM
    {
        int intput; int output;
        ConvLSTMCell clstmc;
        Conv2DLayer conv2D;
        Conv2DLayer conv2D2;
        ConvLSTMCell clstmc5;
        public ConvLSTM()
        {
            //int _intput, int _output
            //intput = _intput; output = _output;
            conv2D = new Conv2DLayer(1, 1, 3, 1, 128);
            clstmc = new ConvLSTMCell(128, 1, 3);
            conv2D2 = new Conv2DLayer(1, 1, 3, 16, 1);
            clstmc5 = new ConvLSTMCell(32, 1, 3);

        }
        TanhLayer sl = new TanhLayer();
        TanhLayer sl2 = new TanhLayer();
        float lr = 0.1f;
        public void train(List<float[][][,]> listdata, List<float[][][,]> listdatay)
        {
            float[][][,] h = null;
            float[][][,] c = null;
            float[][][,] h2 = null;
            float[][][,] c2 = null;
            int len = listdata.Count;
            MSELoss mloss = new MSELoss();
            float loss = 0;
            for (int i = 0; i < len; i++)
            {
                DenseCRF.ImgUtil.savefile(listdata[i][0][0], @"D:\testpng\A" + i + ".png");
                var data= conv2D.Forward(listdata[i]);
                data=sl.Forward(data);
               
                dynamic hh2 = clstmc.Forward(data, h2, c2);
                h2 = hh2.Item1;
                c2 = hh2.Item2;
                //var data2=conv2D2.Forward(h2);
                //data2 = sl2.Forward(data2);
                //var hh = clstmc5.Forward(data2, h, c);
                //h = hh.Item1;
                //c = hh.Item2;
                loss += mloss.Forward(h2, listdatay[i]);
                DenseCRF.ImgUtil.savefile(hh2.Item1[0][0], @"D:\testpng\" + i + ".png");
               
            }
            Console.WriteLine("损失误差："+ loss);
            var gird = mloss.Backward();
           
           // gird= clstmc5.Backward(gird);
            //gird = sl2.Backward(gird);
            //conv2D2.backweight(gird);
            //gird = conv2D2.Backward(gird);
            gird = clstmc.Backward(gird); 
            gird =sl.Backward(gird);
            conv2D.backweight(gird);
            //gird = conv2D.Backward(gird);
          

            clstmc.update(lr);
            conv2D.update(lr);
            //conv2D2.update(lr);
         //   clstmc5.update(lr);
          
        }
        public void test(List<float[][][,]> listdata)
        {
            float[][][,] h = null;
            float[][][,] c = null;
            float[][][,] h2 = null;
            float[][][,] c2 = null;
            int len = listdata.Count;
            // MSELoss mloss = new MSELoss();
            dynamic lostdata=null ;
            float lh = 0;
            //List<float[][][,]> listh = new List<float[][][,]>();
            //List<float[][][,]> listc = new List<float[][][,]>();
            //for (int i = 0; i < 20; i++)
            //{
            //    // DenseCRF.ImgUtil.savefile(listdata[i][0][0], @"D:\testpng\A" + i + ".png");

            //    dynamic data = null;
            //    if (i > 9)
            //    {
            //        data = conv2D.Forward(lostdata);
            //    }
            //    else
            //    {
            //        data = conv2D.Forward(listdata[i]);

            //    }
            //    data = sl.Forward(data);

            //    dynamic hh2 = clstmc.Forward(data, h2, c2);
            //    h2 = hh2.Item1;
            //    c2 = hh2.Item2;
            //    listh.Add(h2);
            //    listc.Add(c2);
            //    lostdata = h2;

            //}
            for (int i = 0; i < 20; i++)
            {
                // DenseCRF.ImgUtil.savefile(listdata[i][0][0], @"D:\testpng\A" + i + ".png");
                var data = conv2D.Forward(listdata[i]);
                data = sl.Forward(data);

                // dynamic hh2 = clstmc.Forward(data, listh[i], listc[i]);
                dynamic hh2 = clstmc.Forward(data, h2, c2);
                h2 = hh2.Item1;
                c2 = hh2.Item2;
                //var data2=conv2D2.Forward(h2);
                //data2 = sl2.Forward(data2);
                //var hh = clstmc5.Forward(data2, h, c);
                //h = hh.Item1;
                //c = hh.Item2;
                //   loss += mloss.Forward(h2, listdatay[i]);
                lostdata = hh2.Item1;
                if (i >= 9)
                {
                    //  listdata.RemoveAt(0);
                    DenseCRF.ImgUtil.savefile(lostdata[0][0], @"D:\testpng\Ba" + lh++ + ".png");
                    listdata.Add(lostdata);
                }

            }
                
              
             
            
           

        }
        public void save(string file)
        {
            List<object> listwb = new List<object>();
            listwb.AddRange(clstmc5.getWB());
            listwb.AddRange(clstmc.getWB());
            listwb.Add(conv2D.weights);
            listwb.Add(conv2D.basicData);
            listwb.Add(conv2D2.weights);
            listwb.Add(conv2D2.basicData);
            string str = Newtonsoft.Json.JsonConvert.SerializeObject(listwb);
            System.IO.StreamWriter sw = new System.IO.StreamWriter(file);
            sw.Write(str);
            sw.Close();
        }
        public void load(string file)
        {
            string str = "";
            System.IO.StreamReader sw = new System.IO.StreamReader(file);
            str = sw.ReadToEnd();
            sw.Close();
            object[] obj = Newtonsoft.Json.JsonConvert.DeserializeObject<object[]>(str);
            clstmc5.load(obj[0], obj[1]);
            clstmc.load(obj[2], obj[3]);
            conv2D.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[4].ToString());
            conv2D.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[5].ToString());
            conv2D2.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[6].ToString());
            conv2D2.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[7].ToString());

        }
    }
}
