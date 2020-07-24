using computational_graph.Layer;
using computational_graph.loss;
using DenseCRF;
using FCN;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
   public class LSTMtest
    {
        static void Main(string[] args)
        {
            单层();
            多层lstm();
            // lstm.Forward()
        }

        static void 单层()
        {
            // float[][][] prev_state = new float[2][][];
            LSTMCELL lstm1 = new LSTMCELL(7, 1);
            LSTMCELL lstm2 = new LSTMCELL(10, 1);
            //var x_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\x_numpy.json"));
            //var h_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\h_numpy.json"));
            //var c_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\c_numpy.json"));
            //var dh_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\dh_numpy.json"));
            var x_numpy = new float[1][];
            var h_numpy = new float[1][];
            var c_numpy = new float[1][];
            var h_numpy2 = new float[1][];
            var c_numpy2 = new float[1][];

            var dh_numpy = new float[1][];
            var dataall = getdata();

            for (int i = 0; i < 30000; i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    h_numpy[j] = new float[1];
                    h_numpy2[j] = new float[1];
                }

                for (int j = 0; j < 1; j++)
                {

                    c_numpy[j] = new float[1];
                    c_numpy2[j] = new float[1];
                }
                var loss = 0.0f;
                MSELoss mloss = new MSELoss();
                for (int j = 0; j < 10; j++)
                {
                    x_numpy[0] = dataall[j + i];

                    var dhgird = lstm1.Forward(x_numpy, h_numpy, c_numpy);

                    h_numpy = dhgird.Item1;
                    c_numpy = dhgird.Item2;
                    dh_numpy[0] = new float[] { dataall[i +j+ 1][0] };
                    loss += mloss.Forward(h_numpy, dh_numpy);

                    var gird = mloss.Backward();
                    lstm1.backward(gird);
                    lstm1.update();

                }
               
                Console.WriteLine("误差:" + loss);

            }
            for (int i = 30000; i < 40000; i++)
            {
                for (int j = 0; j < 1; j++)
                    h_numpy[j] = new float[1];

                for (int j = 0; j < 1; j++)
                    c_numpy[j] = new float[1];
                var loss = 0.0f;
                MSELoss mloss = new MSELoss();

                for (int j = 0; j < 10; j++)
                {
                    x_numpy[0] = dataall[j + i];
                    dh_numpy[0] = new float[] { dataall[j + i + 1][0] };
                    var dhgird = lstm1.Forward(x_numpy, h_numpy, c_numpy);
                    h_numpy = dhgird.Item1;
                    c_numpy = dhgird.Item2;
                    //fff += c_numpy[0][0];


                    //var gird = mloss.Backward();
                    //lstm.backward(gird);
                }
                loss += mloss.Forward(h_numpy, dh_numpy);
                Console.WriteLine("误差:" + loss + ",预测：" + (h_numpy[0][0]) * 1000 + ",期望:" + (dh_numpy[0][0]) * 1000);

            }
        }
      static  void 多层lstm()
        {
            LSTMCELL lstm1 = new LSTMCELL(7, 10);
            LSTMCELL lstm2 = new LSTMCELL(10, 1);
            //var x_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\x_numpy.json"));
            //var h_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\h_numpy.json"));
            //var c_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\c_numpy.json"));
            //var dh_numpy = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\dh_numpy.json"));
            var x_numpy = new float[1][];
            var h_numpy = new float[1][];
            var c_numpy = new float[1][];
            var h_numpy2 = new float[1][];
            var c_numpy2 = new float[1][];

            var dh_numpy = new float[1][];
            var dataall = getdata();

            for (int i = 0; i < 30000; i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    h_numpy[j] = new float[1];
                    h_numpy2[j] = new float[1];
                }

                for (int j = 0; j < 1; j++)
                {

                    c_numpy[j] = new float[10];
                    c_numpy2[j] = new float[1];
                }
                var loss = 0.0f;
                MSELoss mloss = new MSELoss();
                for (int j = 0; j < 10; j++)
                {
                    x_numpy[0] = dataall[j + i];

                    var dhgird = lstm1.Forward(x_numpy, h_numpy, c_numpy);

                    h_numpy = dhgird.Item1;
                    c_numpy = dhgird.Item2;
                    var dhgird2 = lstm2.Forward(h_numpy, h_numpy2, h_numpy2);
                    h_numpy2 = dhgird2.Item1;
                    c_numpy2 = dhgird2.Item2;
                    dh_numpy[0] = new float[] { dataall[i+10][0] };
                    
                }
                loss += mloss.Forward(h_numpy2, dh_numpy);

                var gird = mloss.Backward();
                var gird2 = lstm2.backward(gird);
                lstm1.backward(gird2);
                lstm1.update();
                lstm2.update();
                Console.WriteLine("误差:" + loss);

            }
            for (int i = 30000; i < 40000; i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    h_numpy[j] = new float[1];
                    h_numpy2[j] = new float[1];
                }

                for (int j = 0; j < 1; j++)
                {

                    c_numpy[j] = new float[10];
                    c_numpy2[j] = new float[1];
                }
                var loss = 0.0f;
                MSELoss mloss = new MSELoss();

                for (int j = 0; j < 10; j++)
                {
                    x_numpy[0] = dataall[j + i];
                    dh_numpy[0] = new float[] { dataall[j + i + 1][0] };
                    var dhgird = lstm1.Forward(x_numpy, h_numpy, c_numpy);
                    h_numpy = dhgird.Item1;
                    c_numpy = dhgird.Item2;
                    //fff += c_numpy[0][0];
                    var dhgird2 = lstm2.Forward(h_numpy, h_numpy2, c_numpy2);
                    h_numpy2 = dhgird2.Item1;
                    c_numpy2 = dhgird2.Item2;

                    //var gird = mloss.Backward();
                    //lstm.backward(gird);
                }
                loss += mloss.Forward(h_numpy2, dh_numpy);
                Console.WriteLine("误差:" + loss + ",预测：" + (h_numpy2[0][0]) * 1000 + ",期望:" + (dh_numpy[0][0]) * 1000);

            }
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
    }
   

    public class LSTMCELL
    {
        ConvLayer convLayerih;
        ConvLayer convLayerhh;
        int input_size; int hidden_size;
        public LSTMCELL(int _input_size, int _hidden_size)
        {
            input_size = _input_size;
            hidden_size = _hidden_size;
            convLayerih = new ConvLayer(input_size, hidden_size * 4 );
            //convLayerih.weights = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\lstmihw.json"));
            //convLayerih.basicData = JsonConvert.DeserializeObject<float[]>(util.getstr("D:\\lstmihb.json"));
            convLayerhh = new ConvLayer( hidden_size, hidden_size * 4);
            //convLayerhh.weights = JsonConvert.DeserializeObject<float[][]>(util.getstr("D:\\lstmhhw.json"));
            //convLayerhh.basicData = JsonConvert.DeserializeObject<float[]>(util.getstr("D:\\lstmhhb.json"));
        }
        SigmodLayer input_gate_s = new SigmodLayer();
        SigmodLayer forget_gate_s = new SigmodLayer();
        SigmodLayer output_gate_s = new SigmodLayer();
        TanhLayer cell_memory_tl = new TanhLayer();
        TanhLayer cell_tl = new TanhLayer();
        
        MulLayer c_next_mul = new MulLayer();
        MulLayer mulin_gate_mul = new MulLayer();
        MulLayer h_next_mul = new MulLayer();
        

        public dynamic Forward(float[][] input, float[][] h_prev, float[][] c_prev)
        {
            //a_vector = np.dot(x, self.weight_ih.T) + np.dot(h_prev, self.weight_hh.T)
            //a_vector += self.bias_ih + self.bias_hh
            Xinput = input;
            xh_prev = h_prev;
            xc_prev = c_prev;
            var ih = convLayerih.Forward(input);
            var hh = convLayerhh.Forward(h_prev);
            var a_vector = Matrix.MatrixAdd(ih, hh);
           
            List<float[][]> liast = Matrix.chunk(a_vector,4,1);
            var a_i = liast[0];
            var a_f = liast[1];
            var a_c = liast[2];
            var a_o = liast[3];
          
             input_gate = input_gate_s.Forward(a_i);
             forget_gate = forget_gate_s.Forward(a_f);
             cell_memory = cell_memory_tl.Forward(a_c);
             output_gate = output_gate_s.Forward(a_o);
            var c_next_temp = c_next_mul.Forward(forget_gate, c_prev);
            var mulin_gate = mulin_gate_mul.Forward(input_gate, cell_memory);
            var c_next = Matrix.MatrixAdd(c_next_temp, mulin_gate);

            var h_next = h_next_mul.Forward(output_gate, cell_tl.Forward(c_next));
            
           // dh_prev = Matrix.zroe(h_next.Length, h_next[0].Length);
            return (h_next,c_next);//上次的状态，上次的记忆
        }
        dynamic  Xinput, xh_prev, xc_prev, input_gate, forget_gate, cell_memory, output_gate;
       // dynamic dh_prev;
        dynamic ihweight, hhweight;
        public dynamic backward(dynamic grid)
        {
             
            var dh  = h_next_mul.backwardY(grid);
            var d_tanh_c = cell_tl.Backward(dh);
             //var dc_prev=c_next_mul.backwardY(d_tanh_c);
            

            var d_input_gate = mulin_gate_mul.backward(d_tanh_c);
            var d_forget_gate=c_next_mul.backward(d_tanh_c);
            var d_cell_memory = mulin_gate_mul.backwardY(d_tanh_c);

            var d_output_gate = h_next_mul.backward(grid);// d_tanh_c
            var d_ai = input_gate_s.Backward(d_input_gate);
            var d_af = forget_gate_s.Backward(d_forget_gate);
            var d_ao = output_gate_s.Backward(d_output_gate);
            var d_ac = cell_memory_tl.Backward(d_cell_memory);

            var temp=Matrix.cat(d_ai, d_af, 1);
            var temp2 = Matrix.cat( d_ac, d_ao, 1);
            var da= Matrix.cat(temp, temp2, 1);
           // var daT=Matrix.T(da);
             ihweight = convLayerih.backweight(da);
             hhweight = convLayerhh.backweight(da);
            return convLayerih.backward(da);
        }
        float lr = 0.1f;
        public void update()
        {
            convLayerih.weights = Matrix.MatrixSub(convLayerih.weights, Matrix.multiply(ihweight.grid, lr));
            convLayerih.basicData = Matrix.MatrixSub(convLayerih.basicData, Matrix.multiply(ihweight.basic, lr));

            convLayerhh.weights = Matrix.MatrixSub(convLayerhh.weights, Matrix.multiply(hhweight.grid, lr));
            convLayerhh.basicData = Matrix.MatrixSub(convLayerhh.basicData, Matrix.multiply(hhweight.basic, lr));

        }
    }
}
