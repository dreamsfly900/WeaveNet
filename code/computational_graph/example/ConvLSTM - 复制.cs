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
            float[][][,] datah = new float[1][][,];
            float[][][,] datac = new float[1][][,];
            float[][][,] datay = new float[1][][,];
            datax[0] = new float[10][,];
            datah[0] = new float[10][,];
            datac[0] = new float[10][,];
            datay[0] = new float[10][,];
            MSELoss mloss = new MSELoss();
            ConvLSTM convLSTM = new ConvLSTM(3);
            //  convLSTM.load("radar.bin");
           // convLSTM.load("convLSTM.bin");
            for (int r = 0; r < files.Length - 11; r++)
            {
                
                var loss = 0.0f;
                for (int t = 0; t < 10; t++)
                {
                    string file = files[r + t];
                    float[,] anno1 = DenseCRF.util.readRADARMatrix(file);
                    anno1 = ImgUtil.BilinearInterp(anno1, 128, 128);
                //    ImgUtil.savefile(anno1, "D:/testpng/a" + t + ".png");
                    datax[0][t] = anno1;
                    string file2 = files[r + t + 10];
                    datay[0][t] = DenseCRF.util.readRADARMatrix(file2);
                    datay[0][t] = ImgUtil.BilinearInterp(datay[0][t], 128, 128);
                 //   ImgUtil.savefile(datay[0][t], "D:/testpng/b" + t + ".png");
                }
                while (true)
                {
                    var star = DateTime.Now;
                    var h_next = convLSTM.Forward(datax);

                    for (int ss = 0; ss < 10; ss++)
                        ImgUtil.savefile(h_next[0][ss], "D:/testpng/r" + (r*ss)+ss + ".png");
                    loss = 0f;
                    
                    loss = mloss.Forward(h_next[0][9], datay[0][9]);
                    var gird = mloss.Backward();
                    float[][][,] gird2 = new float[1][][,];
                    gird2[0] = new float[1][,];
                    gird2[0][0] = gird;
                    gird = convLSTM.backward(gird2);
                    Console.WriteLine("lost:" + loss);
                    convLSTM.update();
                    convLSTM.save("convLSTM.bin");
                 
                      
                    var end = DateTime.Now;
                    Console.WriteLine((end - star).TotalMilliseconds);
                }
            }
        }
        
    }
    public class ConvLSTM
    {
        int input_size; int hidden_size; int step=1;
        ConvLSTMCell [] cell;
        Conv2DLayer[] convl;
        ConvTranspose2DLayer[] convt2d;
        computational_graph.Layer.Layer[] tanhLayers;
        List<float[][][,]> listh = new List<float[][][,]>();
        List<float[][][,]> listc = new List<float[][][,]>();
        Averpooling[] pooling = new Averpooling[3];//Averpooling
        public ConvLSTM(int weightssize=3,int _step=1)
        {
            int _input_size = 1; int output = 1,
               //SigmodLayer  TanhLayer
               input_size = _input_size;
            hidden_size = output;
            step = _step;
            cell =new  ConvLSTMCell[3];
            convl = new Conv2DLayer[3];
            convt2d = new ConvTranspose2DLayer[3];
            tanhLayers = new Layer.Layer[6]; 
            convl[0] = new Conv2DLayer(1, 1, weightssize, _input_size, 8, _basic: false);
            pooling[0] = new Averpooling(2);
            cell[0] = new ConvLSTMCell(8, 8, weightssize);
            tanhLayers[0] = new TanhLayer();
            convl[1] = new Conv2DLayer(1, 1, weightssize, 8, 16, _basic: false);
            pooling[1] = new Averpooling(2);
            cell[1] = new ConvLSTMCell(16, 16, weightssize);
            tanhLayers[1] = new TanhLayer();
            convl[2] = new Conv2DLayer(1, 1, weightssize, 16, 32, _basic: false);
            pooling[2] = new Averpooling(2);
            cell[2] = new ConvLSTMCell(32, 32, weightssize);
            tanhLayers[2] = new TanhLayer();
            convt2d[0] = new ConvTranspose2DLayer(2, 1, weightssize+1, 32, 16, _basic: false);
            tanhLayers[3] = new TanhLayer();
            convt2d[1] = new ConvTranspose2DLayer(2, 1, weightssize+1, 16, 8, _basic: false);
            tanhLayers[4] = new TanhLayer();
            convt2d[2] = new ConvTranspose2DLayer(2, 1, weightssize+1, 8, hidden_size, _basic: false);
            tanhLayers[5] = new TanhLayer();

        }
        public dynamic Forward(float[][][,] x)
        {
            float[][][,] h_prev = null;
            float[][][,] c_prev = null;
            float[][][,] h_prev2 = null;
            float[][][,] c_prev2 = null;
            float[][][,] h_prev3 = null;
            float[][][,] c_prev3 = null;
           
          
            float[][][,] prev = new float[1][][,];
            prev[0] = new float[1][,];
            int len = x[0].Length;
            float[][][,] list = new float[1][][,];
            list[0] = new float[len][,];
            
            for (int i = 0; i < len; i++)
            {

                prev[0][0] = x[0][i];
                var y = convl[0].Forward(prev);
                y = tanhLayers[0].Forward(y);
                y=pooling[0].Forward(y);
                if (i == 0)
                {
                    h_prev = zroe(y);
                    c_prev = zroe(y);
                }

                y = cell[0].Forward(y, h_prev, c_prev).Item1;

                y = convl[1].Forward(y);
                y = tanhLayers[1].Forward(y);
                y = pooling[1].Forward(y);
                if (i == 0)
                {
                    h_prev2 = zroe(y);
                    c_prev2 = zroe(y);

                }
                y = cell[1].Forward(y, h_prev2, c_prev2).Item1;

                y = convl[2].Forward(y);
                y = tanhLayers[2].Forward(y);
                y = pooling[2].Forward(y);
                if (i == 0)
                {
                    h_prev3 = zroe(y);
                    c_prev3 = zroe(y);

                }

                y = cell[2].Forward(y, h_prev3, c_prev3).Item1;

                y = convt2d[0].Forward(y);
                y = tanhLayers[3].Forward(y);
                y = convt2d[1].Forward(y);
                y = tanhLayers[4].Forward(y);
                y = convt2d[2].Forward(y);
                y = tanhLayers[5].Forward(y);
                list[0][i] = y[0][0];
            }
            return list;
        }
       
        public dynamic backward(dynamic grid) {


            dynamic grid2= tanhLayers[5].Backward(grid);

             convt2d[2].backweight(grid2);
            grid2 = convt2d[2].Backward(grid2);
          
            grid2 = tanhLayers[4].Backward(grid2);

             convt2d[1].backweight(grid2);
            grid2 = convt2d[1].Backward(grid2);

            grid2 = tanhLayers[3].Backward(grid2);

              convt2d[0].backweight(grid2);
            grid2 = convt2d[0].Backward(grid2);


          

            grid2= cell[2].Backward(grid2);
            grid2= pooling[2].Backward(grid2);
            grid2 = tanhLayers[2].Backward(grid2);
            //cell[2].update();
            convl[2].backweight(grid2);
            grid2 = convl[2].Backward(grid2);

          

            grid2 = cell[1].Backward(grid2);
            grid2 = pooling[1].Backward(grid2);
            grid2 = tanhLayers[1].Backward(grid2);
            convl[1].backweight(grid2);
            grid2 = convl[1].Backward(grid2);

           

            grid2 = cell[0].Backward(grid2);
            grid2 = pooling[0].Backward(grid2);
            grid2 = tanhLayers[0].Backward(grid2);
            convl[0].backweight(grid2);
            grid2 = convl[0].Backward(grid2);
            
            return grid2;
        }
        float lr = 0.5f;
        public void update()
        {
            convt2d[2].update(lr);
            convt2d[1].update(lr);
            convt2d[0].update(lr);
            convl[2].update(lr);
            convl[1].update(lr);
            convl[0].update(lr);
            cell[2].update(lr);
            cell[1].update(lr);
            cell[0].update(lr);

        }
         dynamic zroe(float[][][,] x)
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
        public void load(string file)
        {
            string str = "";
            System.IO.StreamReader sw = new System.IO.StreamReader(file);
            str = sw.ReadToEnd();
            sw.Close();
            List<object> obj = Newtonsoft.Json.JsonConvert.DeserializeObject<List<object>>(str);
            convt2d[2].weights = JsonConvert.DeserializeObject<float[][][,]>(obj[0].ToString());
            convt2d[2].basicData= JsonConvert.DeserializeObject<float[]>(obj[1].ToString());
            convt2d[1].weights = JsonConvert.DeserializeObject<float[][][,]>(obj[2].ToString());
            convt2d[1].basicData = JsonConvert.DeserializeObject<float[]>(obj[3].ToString());
            convt2d[0].weights = JsonConvert.DeserializeObject<float[][][,]>(obj[4].ToString());
            convt2d[0].basicData = JsonConvert.DeserializeObject<float[]>(obj[5].ToString());

            convl[2].weights = JsonConvert.DeserializeObject<float[][][,]>(obj[6].ToString());
            convl[2].basicData = JsonConvert.DeserializeObject<float[]>(obj[7].ToString());
            convl[1].weights = JsonConvert.DeserializeObject<float[][][,]>(obj[8].ToString());
            convl[1].basicData = JsonConvert.DeserializeObject<float[]>(obj[9].ToString());
            convl[0].weights = JsonConvert.DeserializeObject<float[][][,]>(obj[10].ToString());
            convl[0].basicData = JsonConvert.DeserializeObject<float[]>(obj[11].ToString());
            object[] oo = new object[4];
            obj.CopyTo(12, oo, 0, 4);
            cell[2].load(oo);

            oo = new object[4];
            obj.CopyTo(16, oo, 0, 4);
            cell[1].load(oo);

            oo = new object[4];
            obj.CopyTo(20, oo, 0, 4);
            cell[0].load(oo);
        }
        public void save(string file)
        {
            List<object> objlist = new List<object>();
            objlist.Add(convt2d[2].weights);
            objlist.Add(convt2d[2].basicData);
            objlist.Add(convt2d[1].weights);
            objlist.Add(convt2d[1].basicData);
            objlist.Add(convt2d[0].weights);
            objlist.Add(convt2d[0].basicData);
            objlist.Add(convl[2].weights);
            objlist.Add(convl[2].basicData);
            objlist.Add(convl[1].weights);
            objlist.Add(convl[1].basicData);
            objlist.Add(convl[0].weights);
            objlist.Add(convl[0].basicData);
            objlist.AddRange(cell[2].getWB()); 
            objlist.AddRange(cell[1].getWB()); 
            objlist.AddRange(cell[0].getWB());
           
            string str = Newtonsoft.Json.JsonConvert.SerializeObject(objlist);
            System.IO.StreamWriter sw = new System.IO.StreamWriter(file);
            sw.Write(str);
            sw.Close();
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
            convLayerih = new Conv2DLayer(1, (weightssize/2), weightssize, input_size, hidden_size * 4, _basic: false);
           
            convLayerhh = new Conv2DLayer(1, (weightssize / 2), weightssize, hidden_size, hidden_size * 4, _basic: false);
            
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
        public dynamic Backward(dynamic grid)
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
        public void update(float _lr=0.1f )
        {
            lr = _lr;
            convLayerih.weights = Matrix.MatrixSub(convLayerih.weights, Matrix.multiply(ihweight.grid, lr));
            convLayerih.basicData = Matrix.MatrixSub(convLayerih.basicData, Matrix.multiply(ihweight.basic, lr));

            convLayerhh.weights = Matrix.MatrixSub(convLayerhh.weights, Matrix.multiply(hhweight.grid, lr));
            convLayerhh.basicData = Matrix.MatrixSub(convLayerhh.basicData, Matrix.multiply(hhweight.basic, lr));

        }
        public object[] getWB()
        {
            object[] obj = new object[4];
            obj[0] = convLayerih.weights;
            obj[1] = convLayerih.basicData;
            obj[2] = convLayerhh.weights;
            obj[3] = convLayerhh.basicData;
            return obj;
        }
        public void load(object[] obj)
        {
            convLayerih.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[0].ToString());
            convLayerih.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[1].ToString());
            convLayerhh.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[2].ToString());
            convLayerhh.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[3].ToString());
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
