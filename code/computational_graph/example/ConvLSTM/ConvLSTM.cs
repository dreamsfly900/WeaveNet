using computational_graph.loss;
using DenseCRF;
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
            ConvLSTMCell clstmc = new ConvLSTMCell(1, 1, 3);
            float[][][,] x = new float[1][][,];
            x[0] = new float[1][,];
            x[0][0] = new float[5,5];
            x = ones(x);
            float[][][,] y= ones(x);
            float[][][,] h = null;
            float[][][,] c = null;
            MSELoss mloss = new MSELoss();
            var datan = clstmc.Forward(x, h,c);
            var loss= mloss.Forward(datan.Item1, y);
            var gird=  mloss.Backward();
            clstmc.Backward(gird);
            string[] files = System.IO.Directory.GetFiles("test");
            files = files.OrderBy(p => p).ToArray();
            List<float[,]> list = new List<float[,]>();
            for (int r = 0; r < files.Length; r++)
            {
                String file = files[r ];
                float[,] anno1 = DenseCRF.util.readRADARMatrix(file);
                anno1 = ImgUtil.BilinearInterp(anno1, 128, 128);
                list.Add(anno1);
            }
            for (int r = 0; r < list.Count - 10; r++)
            { 

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
    }
    //public class ConvLSTM
    //{
    //    int intput; int output;
    //    List<LSTMCell> list = new List<LSTMCell>();
    //    public ConvLSTM(int _intput,int _output)
    //    {
    //         intput= _intput; output= _output;
    //        for(int i=0;i<)
    //    }
    //}
}
