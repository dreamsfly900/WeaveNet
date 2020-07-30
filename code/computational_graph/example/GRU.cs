using computational_graph.Layer;
using computational_graph.loss;
using FCN;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
    public class GRUtest
    {
        static unsafe void Main(string[] args)
        {
            
            //var star = DateTime.Now;
            //int ss = 0;
            //while (ss < 10)
            //{
            //    float[,] temp = new float[2,2];

            //    float[,] temp2 = new float[2, 2];
            //    fixed (float* arr = &temp[0,0]) 
            //    {
            //        for (int i = 0; i < 2* 2; i++)
            //        {
            //            *(arr + i) = i;
            //            //for (int j = 0; j <= 2; j++)
            //            //     *(&(*(arr + i))+j) = i+j;
            //        }



            //    }
            //    fixed (float* arr = &temp[0, 0])
            //    {
            //        for (int i = 0; i < 2 ; i++)
            //        {

            //            for (int j = 0; j < 2; j++)
            //            {
            //                temp2[i,j] = *(arr + j + (i * 2));
            //            }
            //            //     *(&(*(arr + i))+j) = i+j;
            //        }



            //    }

            //    //for (int i = 0; i < 9000; i++)
            //    //{
            //    //    for (int j = 0; j < 9000; j++)
            //    //    {
            //    //        float f = 0.1f;
            //    //        arr[i, j] = &f;

            //    //    }
            //    //}
            //    // System.Runtime.InteropServices.Marshal.ReleaseComObject(arr);

            //    ss++;
            //}
            //var end = DateTime.Now;
            //Console.WriteLine((end- star).TotalMilliseconds);

            //star = DateTime.Now;
            // ss = 0;
            //while (ss < 10)
            //{
            //    float[,] arr2 = new float[9000, 9000];
            //    for (int i = 0; i < 9000; i++)
            //    {
            //        for (int j = 0; j < 9000; j++)
            //        {

            //            arr2[i, j] = 0.1f;

            //        }
            //    }
            //    ss++;
            //}
            // end = DateTime.Now;
            //Console.WriteLine((end - star).TotalMilliseconds);
            单层();
            // 多层lstm();//多层LSTM 相当于RNN
            // lstm.Forward()
        }
        static float[][] getdata()
        {
            System.IO.StreamReader sr = new System.IO.StreamReader("PRSA_data.txt");

            sr.ReadLine();
            //No,year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir
            //129,-16,-4,1020,SE,1.79,0,0
            List<float[]> datalist = new List<float[]>();
            while (!sr.EndOfStream)
            {
                string str = sr.ReadLine();
                string[] datas = str.Split(',');
                var data = new float[7];
                if (datas[5] == "NA")
                {
                    data[0] = 0;

                }
                else
                    data[0] = (float)Convert.ToDouble(datas[5]) / 1000;
                data[1] = (float)Convert.ToDouble(datas[6]) / 1000;
                data[2] = (float)Convert.ToDouble(datas[7]) / 1000;
                data[3] = (float)Convert.ToDouble(datas[8]) / 1000;
                data[4] = (float)Convert.ToDouble(datas[10]) / 1000;
                data[5] = (float)Convert.ToDouble(datas[11]) / 1000;
                data[6] = (float)Convert.ToDouble(datas[12]) / 1000;
                datalist.Add(data);
            }
            sr.Close();
            return datalist.ToArray();
        }
        static void 单层()
        {
            // float[][][] prev_state = new float[2][][];
            GRU gru = new GRU(7, 15, 1);

            var x_numpy = new float[1][];
            var h_numpy = new float[1][];

            var h_numpy2 = new float[1][];

            var dh_numpy = new float[1][];
            var dataall = getdata();

            for (int i = 0; i < 35000; i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    h_numpy[j] = new float[15];
                    h_numpy2[j] = new float[15];
                }


                var loss = 0.0f;
                MSELoss mloss = new MSELoss();
                for (int j = 0; j < 10; j++)
                {
                    x_numpy[0] = dataall[j + i];

                    var dhgird = gru.Forward(x_numpy, h_numpy);

                    h_numpy = dhgird.Item2;
                    // c_numpy = dhgird.Item2;
                    dh_numpy[0] = new float[] { dataall[i + j + 1][0] };
                    loss += mloss.Forward(dhgird.Item1, dh_numpy);

                    var gird = mloss.Backward();
                    gru.backward(gird);
                    gru.update();

                }

                Console.WriteLine("误差:" + loss);

            }
            for (int i = 35000; i < 40000; i++)
            {
                for (int j = 0; j < 1; j++)
                    h_numpy[j] = new float[15];


                var loss = 0.0f;
                MSELoss mloss = new MSELoss();
                dynamic DY=null;
                for (int j = 0; j < 10; j++)
                {
                    x_numpy[0] = dataall[j + i];
                    dh_numpy[0] = new float[] { dataall[j + i + 1][0] };
                    var dhgird = gru.Forward(x_numpy, h_numpy);
                    h_numpy = dhgird.Item2;
                    DY = dhgird.Item1; 
                   
                }
                loss += mloss.Forward(DY, dh_numpy);
                Console.WriteLine("误差:" + loss + ",预测：" + (DY[0][0]) * 1000 + ",期望:" + (dh_numpy[0][0]) * 1000);

            }
        }
    }
    public class GRU
    {
        private int input_size;
        private int hidden_size;
        private ConvLayer convLayerih;
        private ConvLayer convLayerhh;
        private ConvLayer convLayerhq;
        public GRU(int _input_size, int _hidden_size, int output)
        {
            input_size = _input_size;
            hidden_size = _hidden_size;
            convLayerih = new ConvLayer(input_size, hidden_size * 3);
            convLayerhh = new ConvLayer(hidden_size, hidden_size * 3);
            convLayerhq = new ConvLayer(hidden_size, output);
        }
        SigmodLayer ZSL = new SigmodLayer();
        SigmodLayer RSL = new SigmodLayer();
        TanhLayer THL = new TanhLayer();
        MulLayer RHM = new MulLayer();
        MulLayer ZHL = new MulLayer();
        MulLayer ZHTL = new MulLayer();
        //MulLayer ZHZTL = new MulLayer();
        public (dynamic, dynamic) Forward(float[][] input, float[][] state)
        {
            var H = state;
            var X = convLayerih.Forward(input);
            var HH = convLayerhh.Forward(H);
            List<float[][]> listX = Matrix.chunk(X, 3, 1);
            List<float[][]> listHH = Matrix.chunk(HH, 3, 1);
            var ZS = Matrix.MatrixAdd(listX[0], listHH[0]);
            var Z = ZSL.Forward(ZS);

            var RS = Matrix.MatrixAdd(listX[1], listHH[1]);
            var R = RSL.Forward(RS);

            var RH = RHM.Forward(R, listHH[2]);
            var RHS = Matrix.MatrixAdd(listX[2], RH);
            var H_tilda = THL.Forward(RHS);

            var ZH = ZHL.Forward(Z, H);

            var Z1 = Matrix.MatrixSub(1, Z);
            var ZHT = ZHTL.Forward(Z1, H_tilda);
            H = Matrix.MatrixAdd(ZH, ZHT);
            var Y = convLayerhq.Forward(H);
            return (Y, H);
        }
        dynamic ihweight, hhweight, hqW;
        float lr = 0.1f;
        public dynamic backward(dynamic grid)
        {
              var dy= convLayerhq.backward(grid);
             hqW=convLayerhq.backweight(grid);
            var DH = dy;
            var DZHT=  ZHTL.backward(dy);
           
            var dH_tilda = ZHTL.backwardY(dy);
            //var Dz1 = ZHTL.backward(DH);

            var Dz = ZHL.backward(DZHT);

            var DRHS=THL.Backward(dH_tilda);
            var DR = RHM.backward(DRHS);//Add 因为加法的梯度等于本身 所以DRHS=DRH

            var DHHS = RHM.backwardY(DRHS);
            var DRS = RSL.Backward(DR);
            var DZS = ZSL.Backward(Dz);
            var temp = Matrix.cat(DZS, DRS, 1);
            var da = Matrix.cat(temp, DHHS, 1);
            ihweight = convLayerih.backweight(da);
            hhweight = convLayerhh.backweight(da);
            return convLayerih.backward(da);
        }

        public void update()
        {
            convLayerih.weights = Matrix.MatrixSub(convLayerih.weights, Matrix.multiply(ihweight.grid, lr));
            convLayerih.basicData = Matrix.MatrixSub(convLayerih.basicData, Matrix.multiply(ihweight.basic, lr));

            convLayerhh.weights = Matrix.MatrixSub(convLayerhh.weights, Matrix.multiply(hhweight.grid, lr));
            convLayerhh.basicData = Matrix.MatrixSub(convLayerhh.basicData, Matrix.multiply(hhweight.basic, lr));

            convLayerhq.weights = Matrix.MatrixSub(convLayerhq.weights, Matrix.multiply(hqW.grid, lr));
            convLayerhq.basicData = Matrix.MatrixSub(convLayerhq.basicData, Matrix.multiply(hqW.basic, lr));


        }
    }
}
