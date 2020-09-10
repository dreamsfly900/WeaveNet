using computational_graph.Layer;
using computational_graph.loss;
using DenseCRF;
using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
    public class ConvLSTMtest
    {
        static void Main(string[] args)
        {
            //tian();
          //  test();
           
        }
        static void test()
        {
            string[] files = System.IO.Directory.GetFiles("test");
            files = files.OrderBy(p => p).ToArray();
            float[][][,] datax = new float[1][][,];
            float[][][,] datah = new float[1][][,];
            float[][][,] datac = new float[1][][,];
            float[][][,] datay = new float[1][][,];
            datax[0] = new float[1][,];
            datah[0] = new float[1][,];
            datac[0] = new float[1][,];
            datay[0] = new float[1][,];
            MSELoss mloss = new MSELoss();
            ConvLSTMCell convLSTM = new ConvLSTMCell(1, 1, 5);
            convLSTM.load("radar.bin");
            for (int t = 0; t < 1; t++)
            {
                datah[0][t] = new float[128, 128];
                datac[0][t] = new float[128, 128];
            }
            for (int r = 0; r < files.Length - 11; r++)
            {
                var loss = 0.0f;
                for (int t = 0; t < 10; t++)
                {
                    string file = files[r + t];
                    float[,] anno1 = DenseCRF.util.readRADARMatrix(file);
                    anno1 = ImgUtil.BilinearInterp(anno1, 128, 128);
                    datax[0][0] = anno1;
                    string file2 = files[r + t + 10];
                    datay[0][0] = DenseCRF.util.readRADARMatrix(file2);
                    datay[0][0] = ImgUtil.BilinearInterp(datay[0][0], 128, 128);

                    var (h_next, c_next) = convLSTM.Forward(datax, datah, datac);
                    datah = h_next;
                    datac = c_next;
                    loss += mloss.Forward(h_next, datay);
                     
                        //ImgUtil.savefile(datax[0][t], "testpng/" + (t ) + ".png");
                        //ImgUtil.savefile(datay[0][t], "testpng/" + (t+10) + ".png");
                        ImgUtil.savefile(h_next[0][0], "testpng/a" + (t + 10) + ".png");

                    
                    //var grid = mloss.Backward();
                    //var grid2 = convLSTM.backward(grid);
                    //convLSTM.update();


                }
              
               

                    //var grid = mloss.Backward();
                    //var grid2 = convLSTM.backward(grid);
                    //convLSTM.update();
                    //convLSTM.save("radar.bin");
                //    var end = DateTime.Now;
                //Console.WriteLine((end - star).TotalMilliseconds);
            }
        }
        static void tian()
        {
            var tt = 1;
            string[] files = System.IO.Directory.GetFiles("res");
            files = files.OrderBy(p => p).ToArray();
            float[][][,] datax = new float[1][][,];
            float[][][,] datah = new float[1][][,];
            float[][][,] datac = new float[1][][,];
            float[][][,] datay = new float[1][][,];
            datax[0] = new float[tt][,];
            datah[0] = new float[tt][,];
            datac[0] = new float[tt][,];
            datay[0] = new float[tt][,];
            MSELoss mloss = new MSELoss();
            ConvLSTMCell convLSTM = new ConvLSTMCell(tt, tt, 3);
              convLSTM.load("radar.bin");
            for (int t = 0; t < tt; t++)
            {
                datah[0][t] = new float[128, 128];
                datac[0][t] = new float[128, 128];
            }
            int gg = 0;
            while (gg < 100)
            {
                for (int r = 0; r < 1; r++)
                {
                    var star = DateTime.Now;
                    var loss = 0.0f;
                    for (int t = 0; t < 10; t++)
                    {
                        string file = files[r + t];
                        float[,] anno1 = DenseCRF.util.readRADARMatrix(file);
                        anno1 = ImgUtil.BilinearInterp(anno1, 128, 128);
                        datax[0][0] = anno1;

                        string file2 = files[r + t + 10];
                        datay[0][0] = DenseCRF.util.readRADARMatrix(file2);
                        datay[0][0] = ImgUtil.BilinearInterp(datay[0][0], 128, 128);
                        var (h_next, c_next) = convLSTM.Forward(datax, datah, datac);
                        datah = h_next;
                        datac = c_next;
                        loss += mloss.Forward(h_next, datay);
                        var grid = mloss.Backward();
                        var grid2 = convLSTM.backward(grid);
                        convLSTM.update();
                        ImgUtil.savefile2(h_next[0][0], "testpng/a" + (t + 10) + ".png");
                    }


                
                    Console.WriteLine("误差:" + loss);
                    ////for (int t = 0; t < tt; t++)
                    //ImgUtil.savefile(h_next[0][t], "testpng/a" + (t + 10) + ".png");

                    convLSTM.save("radar.bin");
                    var end = DateTime.Now;
                    Console.WriteLine((end - star).TotalMilliseconds);
                    //for (int t = 0; t < tt; t++)
                    //    ImgUtil.savefile(datah[0][t], "testpng/a" + (t + 10) + ".png");
                }
                gg++;
            }
        }
    }
    public class ConvLSTM
    {
        int input_size; int hidden_size; int step=1;
        ConvLSTMCell [] cell;
        Conv2DLayer[] convl;
        List<float[][][,]> listh = new List<float[][][,]>();
        List<float[][][,]> listc = new List<float[][][,]>();
        public ConvLSTM(int _input_size,int output, int weightssize,int _step)
        {
            input_size = _input_size;
            hidden_size = output;
            step = _step;
            cell =new  ConvLSTMCell[3];
            convl = new Conv2DLayer[3];
           convl[0] = new Conv2DLayer(2, 0, weightssize, _input_size, _input_size);
            cell[0] = new ConvLSTMCell(_input_size, 5, weightssize);
            convl[1] = new Conv2DLayer(2, 0, weightssize, 5, 5);
            cell[1] = new ConvLSTMCell(5, 15, weightssize);
            convl[2] = new Conv2DLayer(2, 0, weightssize, 15, 15);
            cell[2] = new ConvLSTMCell(15, output, weightssize);

        }
        public dynamic Forward(float[][][,] x)
        {
            return 0;
        }

    }
   public class ConvLSTMCell
    {
        Conv2DLayer convLayerih;
        Conv2DLayer convLayerhh;
        int input_size; int hidden_size;
        public ConvLSTMCell(int _input_size, int _hidden_size,int weightssize)
        {
            input_size = _input_size;
            hidden_size = _hidden_size;
            convLayerih = new Conv2DLayer(1, (weightssize/2), weightssize, input_size, hidden_size * 4);
           
            convLayerhh = new Conv2DLayer(1, (weightssize / 2), weightssize, hidden_size, hidden_size * 4);
            
        }
        SigmodLayer input_gate_s = new SigmodLayer();
        SigmodLayer forget_gate_s = new SigmodLayer();
        SigmodLayer output_gate_s = new SigmodLayer();
        TanhLayer cell_memory_tl = new TanhLayer();
        TanhLayer cell_tl = new TanhLayer();

        MulLayer c_next_mul = new MulLayer();
        MulLayer mulin_gate_mul = new MulLayer();
        MulLayer h_next_mul = new MulLayer();


        public (dynamic, dynamic) Forward(float[][][,] input, float[][][,] h_prev, float[][][,] c_prev)
        {
            //a_vector = np.dot(x, self.weight_ih.T) + np.dot(h_prev, self.weight_hh.T)
            //a_vector += self.bias_ih + self.bias_hh
            Xinput = input;
            xh_prev = h_prev;
            xc_prev = c_prev;
            var ih = convLayerih.Forward(input);
            var hh = convLayerhh.Forward(h_prev);
            var a_vector = Matrix.MatrixAdd(ih, hh);

            List<float[][][,]> liast = Matrix.chunk(a_vector, 4, 1);
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
            return (h_next, c_next);//上次的状态，上次的记忆
        }
        dynamic Xinput, xh_prev, xc_prev, input_gate, forget_gate, cell_memory, output_gate;
        // dynamic dh_prev;
        dynamic ihweight, hhweight;
        public dynamic backward(dynamic grid)
        {

            var dh = h_next_mul.backwardY(grid);
            var d_tanh_c = cell_tl.Backward(dh);
            //var dc_prev=c_next_mul.backwardY(d_tanh_c);


            var d_input_gate = mulin_gate_mul.backward(d_tanh_c);
            var d_forget_gate = c_next_mul.backward(d_tanh_c);
            var d_cell_memory = mulin_gate_mul.backwardY(d_tanh_c);

            var d_output_gate = h_next_mul.backward(grid);// d_tanh_c
            var d_ai = input_gate_s.Backward(d_input_gate);
            var d_af = forget_gate_s.Backward(d_forget_gate);
            var d_ao = output_gate_s.Backward(d_output_gate);
            var d_ac = cell_memory_tl.Backward(d_cell_memory);

            var temp = Matrix.cat(d_ai, d_af, 1);
            var temp2 = Matrix.cat(d_ac, d_ao, 1);
            var da = Matrix.cat(temp, temp2, 1);
            // var daT=Matrix.T(da);
            ihweight = convLayerih.backweight(da);
            hhweight = convLayerhh.backweight(da);
            return convLayerih.Backward(da);
        }
        float lr = 0.1f;
        public void update()
        {
            convLayerih.weights = Matrix.MatrixSub(convLayerih.weights, Matrix.multiply(ihweight.grid, lr));
            convLayerih.basicData = Matrix.MatrixSub(convLayerih.basicData, Matrix.multiply(ihweight.basic, lr));

            convLayerhh.weights = Matrix.MatrixSub(convLayerhh.weights, Matrix.multiply(hhweight.grid, lr));
            convLayerhh.basicData = Matrix.MatrixSub(convLayerhh.basicData, Matrix.multiply(hhweight.basic, lr));

        }
        public void load(string file)
        {
            string str = "";
            System.IO.StreamReader sw = new System.IO.StreamReader(file);
            str=sw.ReadToEnd();
            sw.Close();
            object[] obj= Newtonsoft.Json.JsonConvert.DeserializeObject<object[]>(str);
            convLayerih.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[0].ToString()) ;
            convLayerih.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[1].ToString());  
            convLayerhh.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[2].ToString()); 
            convLayerhh.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[3].ToString());

        }
        public void save(string file)
        {
            object[] obj = new object[4];
            obj[0] = convLayerih.weights;
            obj[1] = convLayerih.basicData;
            obj[2] = convLayerhh.weights;
            obj[3] = convLayerhh.basicData;
            string str= Newtonsoft.Json.JsonConvert.SerializeObject(obj);
            System.IO.StreamWriter sw = new System.IO.StreamWriter(file);
            sw.Write(str);
            sw.Close();
        }
    }
}
