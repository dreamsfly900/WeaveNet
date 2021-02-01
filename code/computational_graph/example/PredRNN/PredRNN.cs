using computational_graph.Layer;
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
           // float[][][,] aa = Matrix.zroe2D(1,1,3,3);
           // aa= Matrix.init2Ddata(aa, 1);
           // float[][][,] yy = Matrix.zroe2D(1, 1, 3, 3);
           // yy = Matrix.init2Ddata(yy, 1);

           // Conv2DLayer cl2 = new Conv2DLayer(1, 0, 1, 1, 1);
           // cl2.basicData[0] = 0.5364f;
           // cl2.weights[0][0][0, 0] = -0.0075f  ;
           // MSELoss mSELoss = new MSELoss();

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
            while (true)
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
                    sss = r;
                    predRNN.Forward(listx, listy);
                }
         
          
            //var ww=Gates.backweight(gird);

        }
    }
    public  class PredRNN
    {
        
        PredRNN_Cell rNN_Cell = new PredRNN_Cell(1, 7, 7, 3, true);
        PredRNN_Cell rNN_Cell2 = new PredRNN_Cell(7, 7, 1, 3, true);
        public dynamic Forward(List<float[][][,]> listdata, List<float[][][,]> listdatay)
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
            LeakyReLU leakyRe = new LeakyReLU();
          
            dynamic grids=null, grid;
            for (int i = 0; i < len; i++)
            {
                layer_input_h.Add(new dynamic[2]);
                layer_input_c.Add(new dynamic[2]);
                layer_input_m.Add(new dynamic[2]);
                grid = rNN_Cell.Forward(listdata[i], hs,cs, cm);
                hs = grid.Item1; 
                cs = grid.Item2; 
                cm = grid.Item3;
                layer_input_h[i][0] = hs;
                layer_input_c[i][0] = cs;
                layer_input_m[i][0] = cm;
                grid=rNN_Cell2.Forward(grid.Item1, hs2, cs2, cm2);
                hs2 = grid.Item1;
                cs2 = grid.Item2;
                cm2 = grid.Item3;
                layer_input_h[i][1] = hs2;
                layer_input_c[i][1] = cs2;
                layer_input_m[i][1] = cm2;
            }
            var loss = 0.0;
            for (int i = 0; i < len; i++)
            {
                MSELoss mSELoss = new MSELoss();
                grid = rNN_Cell.Forward(listdata[i], layer_input_h[i][0], layer_input_c[i][0], layer_input_m[i][0]);
                grid = rNN_Cell2.Forward(grid.Item1, layer_input_h[i][1], layer_input_c[i][1], layer_input_m[i][1]);
                grid = leakyRe.Forward(grid.Item1);
                loss+=  mSELoss.Forward(grid, listdatay[i]);
                DenseCRF.ImgUtil.savefile(listdata[i][0][0], @"D:\testpng\A" + PredRNNtest.sss + "a" + i + ".png");
                DenseCRF.ImgUtil.savefile(grid[0][0], @"D:\testpng\B"+ PredRNNtest.sss + "a" + i + ".png");
                grid = mSELoss.Backward();
                if (grids == null)
                    grids = grid;
                else
                    grids= Matrix.MatrixAdd(grids, grid);
            }
            Console.WriteLine("MSE:" + loss);
            grid = leakyRe.Backward(grids);
            grid=rNN_Cell2.Backward(grid);
            grid = rNN_Cell.Backward(grid);

            rNN_Cell.update(0.1f);
            rNN_Cell2.update(0.1f);
            return grid;
        }
    }
}
