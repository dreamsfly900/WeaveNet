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
            
            //var ww=Gates.backweight(gird);
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

        void ceshi()
        {
            double aa = -5.0051e-02;
            Conv2DLayer Gates = new Conv2DLayer(1, (3 / 2), 3, 1, 1, bias: true);
            System.IO.StreamReader sr = new System.IO.StreamReader("Gatesa.json");
            var ss = sr.ReadToEnd();
            sr.Close();
            JObject jsonobj = Newtonsoft.Json.JsonConvert.DeserializeObject<JObject>(ss);
            float[][][,] wi = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(jsonobj["weight"].ToString());
            float[] bais = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(jsonobj["bias"].ToString());
            Gates.weights = wi;
            Gates.basicData = bais;

            ConvLSTMCell clstmc = new ConvLSTMCell(1, 1, 3);
            ConvLSTMCell clstmc2 = new ConvLSTMCell(1, 1, 3);
            float[][][,] x = new float[1][][,];
            x[0] = new float[1][,];
            x[0][0] = new float[5, 5];
            x = ones(x);
            float[][][,] y = ones(x);
            float[][][,] h = null;
            float[][][,] c = null;
            SigmodLayer sf = new SigmodLayer();
            //x=Gates.Forward(x);
            MSELoss mloss = new MSELoss();
            MSELoss mloss2 = new MSELoss();
            float loss = 0;
            float loss2 = 0;
            float[][][,] gird = null;
            float[][][,] gird2 = null;
            for (int s = 0; s < 2; s++)
            {

                var datan = clstmc.Forward(x, h, c);
                h = datan.Item1;
                c = datan.Item2;
                //  var data = Gates.Forward(x);
                //datan = sf.Forward(datan);
                //loss2 += mloss2.Forward(data,y);
                loss += mloss.Forward(datan.Item1, y);
                //if (gird2 is null)
                //    gird2 = mloss2.Backward();
                //else
                //{
                //    gird2 = Matrix.MatrixAdd(gird2, mloss2.Backward());
                //    //  gird = Matrix.MatrixAdd(gird, c);

                //   var gg= Gates.backweight(gird2);
                //}
                // if (gird is null)
                gird = mloss.Backward();
                //else
                {
                    //  gird = Matrix.MatrixAdd(gird, mloss.Backward());
                    //  gird = Matrix.MatrixAdd(gird, c);
                    // gird = mloss.Backward();
                    clstmc.Backward(gird);

                }




            }

            gird = sf.Backward(gird);
            var grad = Gates.backweight(gird);

        }
    }
    public class ConvLSTM
    {
        int intput; int output;
        ConvLSTMCell clstmc;
        ConvLSTMCell clstmc5;
        public ConvLSTM(int _intput, int _output)
        {
            intput = _intput; output = _output;

            clstmc = new ConvLSTMCell(1, 1, 3);
            clstmc5 = new ConvLSTMCell(1, 1, 5);

        }
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
                dynamic hh  =clstmc5.Forward(listdata[i], h, c);
                h = hh.Item1;
                c= hh.Item2;
                dynamic hh2 = clstmc.Forward(hh, h2, c2);
                h2 = hh2.Item1;
                c2 = hh2.Item2;
                loss += mloss.Forward(hh2.Item1, listdatay[i]);
                var gird = mloss.Backward();
                var gird2=clstmc.Backward(gird);
                clstmc5.Backward(gird2);
            }
        }
    }
}
