using computational_graph.Layer;
using FCN;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
    public class ConvLSTMCell
    {
        Conv2DLayer Gates;
       
        int input_size; int hidden_size;
        public ConvLSTMCell(int _input_size, int _hidden_size, int weightssize)
        {
            input_size = _input_size;
            hidden_size = _hidden_size;
            Gates = new Conv2DLayer(1, (weightssize / 2), weightssize, input_size + hidden_size, 4 * hidden_size, bias: true);


         

            System.IO.StreamReader sr = new System.IO.StreamReader("Gates.json");
            var ss = sr.ReadToEnd();
            sr.Close();
            JObject jsonobj = Newtonsoft.Json.JsonConvert.DeserializeObject<JObject>(ss);
            float[][][,] wi = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(jsonobj["weight"].ToString());
            float[] bais = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(jsonobj["bias"].ToString());
            Gates.weights = wi;
            Gates.basicData = bais;

        }
        SigmodLayer input_gate_s = new SigmodLayer();
        SigmodLayer forget_gate_s = new SigmodLayer();
        SigmodLayer output_gate_s = new SigmodLayer();
        TanhLayer cell_gate_tl = new TanhLayer();
        TanhLayer cell_tl = new TanhLayer();

        MulLayer c_next_mul = new MulLayer();
        MulLayer mulin_gate_mul = new MulLayer();
        MulLayer h_next_mul = new MulLayer();

        public dynamic grad;
        public (dynamic, dynamic) Forward(float[][][,] input, float[][][,] prev_hidden, float[][][,] prev_cell)
        {
            //a_vector = np.dot(x, self.weight_ih.T) + np.dot(h_prev, self.weight_hh.T)
            //a_vector += self.bias_ih + self.bias_hh

            if (prev_hidden is null)
            {
                prev_hidden = new float[input.Length][][,];
                prev_cell = new float[input.Length][][,];
                for (int i = 0; i < input.Length; i++)
                {
                    prev_hidden[i] = new float[hidden_size][,];
                    prev_cell[i] = new float[hidden_size][,];
                    for (int j = 0; j < hidden_size; j++)
                    {
                        prev_hidden[i][j] = new float[input[0][0].GetLength(0), input[0][0].GetLength(1)];
                        prev_cell[i][j] = new float[input[0][0].GetLength(0), input[0][0].GetLength(1)];
                    }
                }
            }
            //else {
            //    prev_hidden = copy(prev_hidden);
            //    prev_cell = copy(prev_cell);
            //}

            var stacked_inputs=  Matrix.cat(input, prev_hidden, 1);
            var gates = Gates.Forward(stacked_inputs);
            List<float[][][,]>  dynamics= Matrix.chunk(gates, 4, 1);
            dynamic in_gate = dynamics[0];
            dynamic remember_gate = dynamics[1];
            dynamic out_gate = dynamics[2];
            dynamic cell_gate = dynamics[3];
            //三个门
            in_gate=input_gate_s.Forward(in_gate);
            remember_gate=forget_gate_s.Forward(remember_gate);
            out_gate=output_gate_s.Forward(out_gate);
            //遗忘门？
            cell_gate=cell_gate_tl.Forward(cell_gate);
            var cell=Matrix.MatrixAdd(   mulin_gate_mul.Forward(remember_gate, prev_cell), c_next_mul.Forward(in_gate, cell_gate));
           var  hidden=h_next_mul.Forward(out_gate,cell_tl.Forward(cell));
            // dh_prev = Matrix.zroe(h_next.Length, h_next[0].Length);
            return (hidden, cell);//上次的状态，上次的记忆
        }
        dynamic copy(float[][][,]  input)
        {
             float[][][,]  prev_hidden = new float[input.Length][][,];
            
            for (int i = 0; i < input.Length; i++)
            {
                prev_hidden[i] = new float[hidden_size][,];
                 
                for (int j = 0; j < hidden_size; j++)
                {
                    prev_hidden[i][j] = new float[input[0][0].GetLength(0), input[0][0].GetLength(1)];
                    for (int a = 0; a < input[0][0].GetLength(0); a++)
                        for (int b = 0; b < input[0][0].GetLength(1); b++)
                            prev_hidden[i][j][a, b] = input[i][j][a, b];
                }
            }
            return prev_hidden;
        }
        dynamic griddata;
        public dynamic Backward(dynamic grid)
        {
            dynamic out_gate = h_next_mul.Backward(grid);
            out_gate=output_gate_s.Backward(out_gate);
            
            

            dynamic cell= h_next_mul.BackwardY(grid);
            cell = cell_tl.Backward(cell);
            //var prev_cell = mulin_gate_mul.BackwardY(cell);
            dynamic remember_gate = mulin_gate_mul.Backward(cell);
            remember_gate=forget_gate_s.Backward(remember_gate);
     
            
            

            dynamic in_gate = c_next_mul.Backward(cell);
            in_gate=input_gate_s.Backward(in_gate);

        
              
            

            dynamic cell_gate = c_next_mul.BackwardY(cell);
            cell_gate=cell_gate_tl.Backward(cell_gate);

          
              
           

            var ir= Matrix.cat(in_gate, remember_gate, 1);
            var iro = Matrix.cat(ir, out_gate, 1);
            var grid2 = Matrix.cat(iro, cell_gate, 1);
            //if (griddata == null)
                griddata = grid2;
            //else
            //    griddata = Matrix.MatrixAdd(griddata, grid2);
            if (grad == null)
                grad = Gates.backweight(griddata);
            else {
                var temp = Gates.backweight(griddata);
                grad=  new { grid = Matrix.MatrixAdd(grad.grid, temp.grid), basic = Matrix.MatrixAdd(grad.basic, temp.basic) };
              
            }
            
            grid = Gates.Backward(grid2);

            return Matrix.chunk(grid, 2, 1)[0];
        }
        public void update(dynamic lr)
        {
            Gates.grid = grad;
            Gates.update(lr);
        }
    }

    //public class ConvLSTMCell
    //{
    //    Conv2DLayer convLayerih;
    //    Conv2DLayer convLayerhh;
    //    int input_size; int hidden_size;
    //    public ConvLSTMCell(int _input_size, int _hidden_size, int weightssize)
    //    {
    //        input_size = _input_size;
    //        hidden_size = _hidden_size;
    //        convLayerih = new Conv2DLayer(1, (weightssize / 2), weightssize, input_size, hidden_size * 4, bias: true);
            

    //         convLayerhh = new Conv2DLayer(1, (weightssize / 2), weightssize, hidden_size, hidden_size * 4, bias: false);

    //        System.IO.StreamReader sr = new System.IO.StreamReader("Gates.json");
    //        var ss = sr.ReadToEnd();
    //        sr.Close();
    //        JObject jsonobj = Newtonsoft.Json.JsonConvert.DeserializeObject<JObject>(ss);
    //        float[][][,] wi = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(jsonobj["weight"].ToString());
    //        float[] bais = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(jsonobj["bias"].ToString());
    //        for (int i = 0; i < 4; i++)
    //        {
    //            convLayerih.weights[i][0] = wi[i][0];
    //            convLayerih.basicData[i] = bais[i];
    //            convLayerhh.weights[i][0] = wi[i][1];
    //        }

    //    }
    //    SigmodLayer input_gate_s = new SigmodLayer();
    //    SigmodLayer forget_gate_s = new SigmodLayer();
    //    SigmodLayer output_gate_s = new SigmodLayer();
    //    TanhLayer cell_memory_tl = new TanhLayer();
    //    TanhLayer cell_tl = new TanhLayer();

    //    MulLayer c_next_mul = new MulLayer();
    //    MulLayer mulin_gate_mul = new MulLayer();
    //    MulLayer h_next_mul = new MulLayer();


    //    public (dynamic, dynamic) Forward(float[][][,] input, float[][][,] h_prev, float[][][,] c_prev)
    //    {
    //        //a_vector = np.dot(x, self.weight_ih.T) + np.dot(h_prev, self.weight_hh.T)
    //        //a_vector += self.bias_ih + self.bias_hh
    //        Xinput = input;
    //        xh_prev = h_prev;
    //        xc_prev = c_prev;
    //        var ih = convLayerih.Forward(input);
    //        var hh = convLayerhh.Forward(h_prev);
    //        var a_vector = Matrix.MatrixAdd(ih, hh);

    //        List<float[][][,]> liast = Matrix.chunk(a_vector, 4, 1);
    //        var a_i = liast[0];
    //        var a_f = liast[1];
    //        var a_c = liast[2];
    //        var a_o = liast[3];

    //        input_gate = input_gate_s.Forward(a_i);
    //        forget_gate = forget_gate_s.Forward(a_f);
    //        cell_memory = cell_memory_tl.Forward(a_c);
    //        output_gate = output_gate_s.Forward(a_o);
    //        var c_next_temp = c_next_mul.Forward(forget_gate, c_prev);
    //        var mulin_gate = mulin_gate_mul.Forward(input_gate, cell_memory);
    //        var c_next = Matrix.MatrixAdd(c_next_temp, mulin_gate);

    //        var h_next = h_next_mul.Forward(output_gate, cell_tl.Forward(c_next));

    //        // dh_prev = Matrix.zroe(h_next.Length, h_next[0].Length);
    //        return (h_next, c_next);//上次的状态，上次的记忆
    //    }
    //    dynamic Xinput, xh_prev, xc_prev, input_gate, forget_gate, cell_memory, output_gate;
    //    // dynamic dh_prev;
    //    dynamic ihweight, hhweight;
    //    public dynamic Backward(dynamic grid)
    //    {

    //        var dh = h_next_mul.backwardY(grid);
    //        var d_tanh_c = cell_tl.Backward(dh);
    //        //var dc_prev=c_next_mul.backwardY(d_tanh_c);


    //        var d_input_gate = mulin_gate_mul.backward(d_tanh_c);
    //        var d_forget_gate = c_next_mul.backward(d_tanh_c);
    //        var d_cell_memory = mulin_gate_mul.backwardY(d_tanh_c);

    //        var d_output_gate = h_next_mul.backward(grid);// d_tanh_c
    //        var d_ai = input_gate_s.Backward(d_input_gate);
    //        var d_af = forget_gate_s.Backward(d_forget_gate);
    //        var d_ao = output_gate_s.Backward(d_output_gate);
    //        var d_ac = cell_memory_tl.Backward(d_cell_memory);

    //        var temp = Matrix.cat(d_ai, d_af, 1);
    //        var temp2 = Matrix.cat(d_ac, d_ao, 1);
    //        var da = Matrix.cat(temp, temp2, 1);
    //        // var daT=Matrix.T(da);
    //        ihweight = convLayerih.backweight(da);
    //        hhweight = convLayerhh.backweight(da);
    //        return convLayerih.Backward(da);
    //    }
    //    float lr = 0.1f;
    //    public void update(float _lr = 0.1f)
    //    {
    //        lr = _lr;
    //        convLayerih.weights = Matrix.MatrixSub(convLayerih.weights, Matrix.multiply(ihweight.grid, lr));
    //        convLayerih.basicData = Matrix.MatrixSub(convLayerih.basicData, Matrix.multiply(ihweight.basic, lr));

    //        convLayerhh.weights = Matrix.MatrixSub(convLayerhh.weights, Matrix.multiply(hhweight.grid, lr));
    //        convLayerhh.basicData = Matrix.MatrixSub(convLayerhh.basicData, Matrix.multiply(hhweight.basic, lr));

    //    }
    //    public object[] getWB()
    //    {
    //        object[] obj = new object[4];
    //        obj[0] = convLayerih.weights;
    //        obj[1] = convLayerih.basicData;
    //        obj[2] = convLayerhh.weights;
    //        obj[3] = convLayerhh.basicData;
    //        return obj;
    //    }
    //    public void load(object[] obj)
    //    {
    //        convLayerih.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[0].ToString());
    //        convLayerih.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[1].ToString());
    //        convLayerhh.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[2].ToString());
    //        convLayerhh.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[3].ToString());
    //    }
    //    public void load(string file)
    //    {
    //        string str = "";
    //        System.IO.StreamReader sw = new System.IO.StreamReader(file);
    //        str = sw.ReadToEnd();
    //        sw.Close();
    //        object[] obj = Newtonsoft.Json.JsonConvert.DeserializeObject<object[]>(str);
    //        convLayerih.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[0].ToString());
    //        convLayerih.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[1].ToString());
    //        convLayerhh.weights = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][,]>(obj[2].ToString());
    //        convLayerhh.basicData = Newtonsoft.Json.JsonConvert.DeserializeObject<float[]>(obj[3].ToString());

    //    }
    //    public void save(string file)
    //    {
    //        object[] obj = new object[4];
    //        obj[0] = convLayerih.weights;
    //        obj[1] = convLayerih.basicData;
    //        obj[2] = convLayerhh.weights;
    //        obj[3] = convLayerhh.basicData;
    //        string str = Newtonsoft.Json.JsonConvert.SerializeObject(obj);
    //        System.IO.StreamWriter sw = new System.IO.StreamWriter(file);
    //        sw.Write(str);
    //        sw.Close();
    //    }
    //}

    //public class LSTMCell
    //{
    //    int input_size, hidden_size;
    //    Conv2DLayer Wxi, Whi, Wxf, Whf, Wxc, Whc, Wxo, Who;
    //    dynamic Wci, Wcf, Wco;
    //    public LSTMCell(int _input_size, int _hidden_size, int weightssize)
    //    {
    //        input_size = _input_size;
    //        hidden_size = _hidden_size;
    //        var padding = (weightssize - 1) / 2;
    //        Wxi = new Conv2DLayer(1, padding, in_channels: input_size, out_channels: hidden_size, kernel_size: weightssize, bias: true);
    //        Whi = new Conv2DLayer(1, padding, in_channels: hidden_size, out_channels: hidden_size, kernel_size: weightssize, bias: false);

    //        Wxf = new Conv2DLayer(1, padding, in_channels: input_size, out_channels: hidden_size, kernel_size: weightssize, bias: true);
    //        Whf = new Conv2DLayer(1, padding, in_channels: hidden_size, out_channels: hidden_size, kernel_size: weightssize, bias: false);
    //        Wxc = new Conv2DLayer(1, padding, in_channels: input_size, out_channels: hidden_size, kernel_size: weightssize, bias: true);
    //        Whc = new Conv2DLayer(1, padding, in_channels: hidden_size, out_channels: hidden_size, kernel_size: weightssize, bias: false);
    //        Wxo = new Conv2DLayer(1, padding, in_channels: input_size, out_channels: hidden_size, kernel_size: weightssize, bias: true);
    //        Who = new Conv2DLayer(1, padding, in_channels: hidden_size, out_channels: hidden_size, kernel_size: weightssize, bias: false);

    //    }
      
    //    SigmodLayer sl = new SigmodLayer();
    //    SigmodLayer sl2 = new SigmodLayer();
    //    SigmodLayer sl3 = new SigmodLayer();
    //    TanhLayer th = new TanhLayer();
    //    TanhLayer th2 = new TanhLayer();
    //    MulLayer cWcimul = new MulLayer();
    //    MulLayer cWcfmul = new MulLayer();
    //    MulLayer cWcomul = new MulLayer();
    //    MulLayer cfcmul = new MulLayer();
    //    MulLayer cithxcmul = new MulLayer();
    //    MulLayer new_hmul = new MulLayer();
        

    //    public (dynamic, (dynamic, dynamic)) Forward(float[][][,] input, (dynamic, dynamic) hidden_state)
    //    {
    //        dynamic hidden = hidden_state.Item1;
    //        dynamic c = hidden_state.Item2;

    //        //hidden, c = hidden_state  # hidden and c are images with several channels
    //        if (Wci == null)
    //        {
    //            Wci = c;
    //            Wcf = c;
    //            Wcf = c;
    //        }
    //        var Wxiconv = Wxi.Forward(input);
    //        var Whiconv = Whi.Forward(hidden);
            
    //        var cWci = cWcimul.Forward(c, Wci);
    //        var sumwwc = Matrix.MatrixAdd(Matrix.MatrixAdd(Wxiconv, Whiconv), cWci);
    //        var ci = sl.Forward(sumwwc);
    //        //ci = torch.sigmoid(self.Wxi(input) + self.Whi(hidden) + c * self.Wci)
    //        var Wxfconv = Wxf.Forward(input);
    //        var Whfconv = Whf.Forward(hidden);
           
    //        var cWcf = cWcfmul.Forward(c, Wcf);
    //        var sumcf = Matrix.MatrixAdd(Matrix.MatrixAdd(Wxfconv, Whfconv), cWcf);
    //        var cf = sl2.Forward(sumcf);
    //        //cf = torch.sigmoid(self.Wxf(input) + self.Whf(hidden) + c * self.Wcf)
    //        var Wxoconv = Wxo.Forward(input);
    //        var Whoconv = Who.Forward(hidden);
           
    //        var cWco = cWcomul.Forward(c, Wco);
    //        var sumco = Matrix.MatrixAdd(Matrix.MatrixAdd(Wxoconv, Whoconv), cWco);
    //        var co = sl3.Forward(sumco);
    //        //co = torch.sigmoid(self.Wxo(input) + self.Who(hidden) + c * self.Wco)
    //        var Wxccov = Wxc.Forward(input);
    //        var Whccov = Whc.Forward(hidden);

    //        var thxc = th.Forward(Matrix.MatrixAdd(Wxccov, Whccov));
    //        var cfc = cfcmul.Forward(cf, c);
    //        var cithxc = cithxcmul.Forward(ci, thxc);
    //        var new_c = Matrix.MatrixAdd(cfc, cithxc);
    //        //new_c = cf * c + ci * torch.tanh(self.Wxc(input) + self.Whc(hidden))

    //        var new_h = new_hmul.Forward(co, th2.Forward(new_c));
    //        //new_h = co * torch.tanh(new_c)
    //        return (new_h, (new_h, new_c));
    //    }
    //    dynamic zroe(float[][][,] x)
    //    {
    //        float[][][,] h_prev = new float[x.Length][][,];
    //        for (int i = 0; i < x.Length; i++)
    //        {
    //            h_prev[i] = new float[x[i].Length][,];

    //            for (int j = 0; j < x[i].Length; j++)
    //            {
    //                h_prev[i][j] = new float[x[i][j].GetLength(0), x[i][j].GetLength(1)];

    //            }
    //        }
    //        return h_prev;
    //    }
    //    public dynamic backward(dynamic grid)
    //    {
    //        dynamic girdret;
    //       var grid2=new_hmul.backwardY(grid);
    //       var cogrid2 = new_hmul.backward(grid);
    //        var sumcogrid = sl3.Backward(cogrid2);
    //        Wxo.backweight(sumcogrid);
    //        Who.backweight(sumcogrid);

    //        var grid2new_c=th2.Backward(grid2);
    //        var thxcgird = cithxcmul.backwardY(grid2new_c);
    //       var cigrid= cithxcmul.backward(grid2new_c);
    //        var sumwwcgrid = sl.Backward(cigrid);
    //        Wxi.backweight(sumwwcgrid);
    //        Whi.backweight(sumwwcgrid);

           

    //       var Whcgrid= th.Backward(thxcgird);
    //       Wxc.backweight(Whcgrid); 
    //       Whc.backweight(Whcgrid);

    //      //  var inputgird = Wxc.Backward(Whcgrid);
    //        var cfgird=cfcmul.backward(grid2new_c);
    //       var sumcfgrid= sl2.Backward(cfgird);
    //        Wxf.backweight(sumcfgrid);
    //        Whf.backweight(sumcfgrid);

    //        girdret = Wxi.Backward(sumwwcgrid);
    //        //  var cgird = cfcmul.backwardY(grid2new_c);
    //        return girdret;
    //    }
    //}
}
