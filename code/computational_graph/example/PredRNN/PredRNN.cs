﻿using computational_graph.Layer;
using computational_graph.loss;
using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example.PredRNN
{
    public class PredRNNtest
    {
       public   static int sss = 0;
        static void Main(string[] args)
        {
           



            MSELoss mSELoss = new MSELoss();

            // dynamic grids=null;
            // for (int i = 0; i < 2; i++)
            // {
            //     var da1 = cl2.Forward(aa);
            //     float lost = mSELoss.Forward(da1, yy);
            //     var gird = mSELoss.Backward();
            //     if (grids == null)
            //         grids = gird;
            //     else
            //         grids = Matrix.MatrixAdd(grids, gird);

            // }
            //var gir = cl2.backweight(grids);

            ///train();
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
            PredRNN predRNN = new PredRNN();
           //  predRNN.load("predRNN.bin");
            while (true)
            {
                for (int r = 0; r < list.Count - (10); r++)
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
                    sss = r;
                    //while (true)
                    predRNN.tian(listx, listy);
                  //  predRNN.test(listx);
                  //  predRNN.save("predRNN.bin");
                }
              
            }
         
          
            //var ww=Gates.backweight(gird);

        }
    }
    public  class PredRNN
    {

       // PredRNN_Cell rNN_Cell = new PredRNN_Cell(1, 7, 7, 3, true);
       //PredRNN_Cell rNN_Cell2 = new PredRNN_Cell(7, 7, 1, 3, true);
     //   Conv2DLayer conv2D = new Conv2DLayer(1, 1, 3, 1, 64);
        PredRNN_Cell rNN_Cell3 = new PredRNN_Cell(1, 7, 7, 3, true);
        PredRNN_Cell rNN_Cell4 = new PredRNN_Cell(7, 7, 1, 3, true);
        //LeakyReLU leakyRe = new LeakyReLU();
        //LeakyReLU leakyRe2 = new LeakyReLU();
        public void tian(List<float[][][,]> listdata, List<float[][][,]> listdatay)
        {
            Matrix.CUDA = true;
            float[][][,] hs = null;
            float[][][,] cs = null;
            float[][][,] cm = null;
            float[][][,] hs2 = null;
            float[][][,] cs2 = null;
            float[][][,] cm2 = null;
            int len = listdata.Count;
            List<dynamic[]> layer_input_h = new List<dynamic[]>();
            List<dynamic[]> layer_input_c = new List<dynamic[]>();
            List<dynamic[]> layer_input_m = new List<dynamic[]>();
          
            var loss = 0.0;
            dynamic grids=null, grids2=null, grid=null;
            for (int i = 0; i < 1; i++)
            {

                //   layer_input_m.Add(new dynamic[2]);
                //grid = conv2D.Forward(listdata[i]);
                //grid = leakyRe2.Forward(grid);
                DateTime stat = DateTime.Now;
                grid = rNN_Cell3.Forward(listdata[i], hs, cs, cm);
                DateTime end = DateTime.Now;
                Console.WriteLine($"计算时间：{(end - stat).TotalMilliseconds}");
                hs = grid.Item1;
                cs = grid.Item2;
                cm = grid.Item3;
                grid = rNN_Cell4.Forward(grid.Item1, hs2, cs2, cm2);
                hs2 = grid.Item1;
                cs2 = grid.Item2;
                cm2 = grid.Item3;
            }
            MSELoss mSELoss = new MSELoss();
            for (int i = 0; i < len; i++)
            {
             
              
                grid = rNN_Cell3.Forward(listdata[i], hs,cs, cm);
                hs = grid.Item1; 
                cs = grid.Item2; 
                cm = grid.Item3;
                //layer_input_h[i][0] = hs;
                //layer_input_c[i][0] = cs;
                //layer_input_m[i][0] = cm;
                grid = rNN_Cell4.Forward(grid.Item1, hs2, cs2, cm2);
                hs2 = grid.Item1;
                cs2 = grid.Item2;
                cm2 = grid.Item3;
                //layer_input_h[i][1] = hs2;
                //layer_input_c[i][1] = cs2;
                //layer_input_m[i][1] = cm2;
                //  grid = leakyRe.Forward(grid.Item1);
                //DenseCRF.ImgUtil.savefile(listdata[i][0][0], @"D:\testpng\A" + PredRNNtest.sss + "a" + i + ".png");
                //  DenseCRF.ImgUtil.savefile(listdatay[i][0][0], @"D:\testpng\B" + PredRNNtest.sss + "a" + i + ".png");
                DenseCRF.ImgUtil.savefile(hs[0][0], @"D:\testpng\B" + PredRNNtest.sss + "y" + i + ".png");

                loss += mSELoss.Forward(grid.Item1, listdatay[i]);
                Console.WriteLine("MSE:" + loss);
                grid = mSELoss.Backward();

                //  grids2 = grid;
                if (grids2 == null)
                    grids2 = grid;
                else
                    grids2 = Matrix.MatrixAdd(grids2, grid);
            }



          

           // return grid;
        }
        public void test(List<float[][][,]> listdata)
        {

            float[][][,] hs = null;
            float[][][,] cs = null;
            float[][][,] cm = null;
            float[][][,] hs2 = null;
            float[][][,] cs2 = null;
            float[][][,] cm2 = null;
            int len = listdata.Count;
            List<dynamic[]> layer_input_h = new List<dynamic[]>();
            List<dynamic[]> layer_input_c = new List<dynamic[]>();
            List<dynamic[]> layer_input_m = new List<dynamic[]>();

          dynamic grid=null;
            //for (int i = 0; i < 1; i++)
            //{

            //    //   layer_input_m.Add(new dynamic[2]);
            //    grid = conv2D.Forward(listdata[i]);
            //    grid = leakyRe2.Forward(grid);
            //    grid = rNN_Cell3.Forward(grid, hs, cs, cm);
            //    hs = grid.Item1;
            //    cs = grid.Item2;
            //    cm = grid.Item3;
            //    //grid = rNN_Cell4.Forward(grid.Item1, hs2, cs2, cm2);
            //    //hs2 = grid.Item1;
            //    //cs2 = grid.Item2;
            //    //cm2 = grid.Item3;
            //}
           
            for (int i = 0; i < len+10; i++)
            {

                //layer_input_h.Add(new dynamic[2]);
                //layer_input_c.Add(new dynamic[2]);
                //layer_input_m.Add(new dynamic[2]);
                //grid = conv2D.Forward(listdata[i]);
                //grid = leakyRe2.Forward(grid);
                grid = rNN_Cell3.Forward(grid, hs, cs, cm);
                hs = grid.Item1;
                cs = grid.Item2;
                cm = grid.Item3;
                //layer_input_h[i][0] = hs;
                //layer_input_c[i][0] = cs;
               // layer_input_m[i][0] = cm;
                //grid= rNN_Cell4.Forward(grid.Item1, hs2, cs2, cm2);
                //hs2 = grid.Item1;
                //cs2 = grid.Item2;
                //cm2 = grid.Item3;
                //layer_input_h[i][1] = hs2;
                //layer_input_c[i][1] = cs2;
                //layer_input_m[i][1] = cm2;
                 // grid = leakyRe.Forward(grid.Item1);
                if (i >= 9)
                {
                    listdata.Add(hs);
                    DenseCRF.ImgUtil.savefile(grid.Item1[0][0], @"D:\testpng\B" + PredRNNtest.sss + "a" + i + ".png");
                }
                else
                DenseCRF.ImgUtil.savefile(listdata[i][0][0], @"D:\testpng\A" + PredRNNtest.sss + "a" + i + ".png");

              
  
                //  grids2 = grid;
                //if (grids2 == null)
                //    grids2 = grid;
                //else
                //    grids2 = Matrix.MatrixAdd(grids2, grid);
            }



        }
        public void save(String file)
        {
            List<object> listwb = new List<object>();
            listwb.AddRange(rNN_Cell3.getWB()); 
            //listwb.Add(conv2D.weights);
            //listwb.Add(conv2D.basicData);
           
            string str = Newtonsoft.Json.JsonConvert.SerializeObject(listwb);
            System.IO.StreamWriter sw = new System.IO.StreamWriter(file);
            sw.Write(str);
            sw.Close(); 
        }
        public void load(string file)
        {
            try
            {
                string str = "";
                System.IO.StreamReader sw = new System.IO.StreamReader(file);
                str = sw.ReadToEnd();
                sw.Close();
                object[] obj = Newtonsoft.Json.JsonConvert.DeserializeObject<object[]>(str);
                rNN_Cell3.load(obj);
                //conv2D.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[8].ToString());
                //conv2D.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[9].ToString());
            }
            catch { }

        }
    }
}
