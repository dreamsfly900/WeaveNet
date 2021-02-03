using computational_graph.Layer;
using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example.PredRNN
{
   public class PredRNN_Cell
    {
        Conv2DLayer conv_h_c, conv_m, conv_o, conv_h_next;
        int hidden_dim; int hidden_dim_m, input_dim;

        public PredRNN_Cell(int  input_dim,int  hidden_dim_m,int  hidden_dim,int kernel_size,bool bias)
        {
            this.input_dim = input_dim;
              this.hidden_dim= hidden_dim;
            this.hidden_dim_m = hidden_dim_m;
            int padding = kernel_size / 2;
             conv_h_c = new Conv2DLayer(1, padding, kernel_size: kernel_size, input_dim + hidden_dim,
                                  out_channels : 3 * hidden_dim,
                                  bias : bias);
             conv_m = new Conv2DLayer(1, padding, kernel_size: kernel_size, in_channels: input_dim + hidden_dim_m,
                                  out_channels: 3 * hidden_dim_m,
                                  bias: bias);
             conv_o = new Conv2DLayer(1, padding, kernel_size: kernel_size, input_dim + hidden_dim * 2 + hidden_dim_m,
                                  out_channels: hidden_dim,
                                  bias: bias);
             conv_h_next = new Conv2DLayer(1, padding, kernel_size: kernel_size,
               in_channels: hidden_dim + hidden_dim_m,
                                out_channels: hidden_dim,
                                bias: bias);



        }
        SigmodLayer sli = new SigmodLayer();
        SigmodLayer slf = new SigmodLayer();
        TanhLayer slg = new TanhLayer();
        MulLayer mulf = new MulLayer();
        MulLayer mulig = new MulLayer();
        MulLayer mulf_1 = new MulLayer();
        MulLayer mulig_1 = new MulLayer();
        SigmodLayer sli_m = new SigmodLayer();
        SigmodLayer slf_m = new SigmodLayer();
        TanhLayer slg_m = new TanhLayer();
        SigmodLayer slcombined_o = new SigmodLayer();
        TanhLayer h_next_tl = new TanhLayer();
        MulLayer mulo = new MulLayer();
        public dynamic Forward(dynamic input_tensor,dynamic h_cur, dynamic cur_state, dynamic cur_state_m)
        {
            if (cur_state is null)
            {
                h_cur = new float[input_tensor.Length][][,];
                cur_state = new float[input_tensor.Length][][,];
                cur_state_m = new float[input_tensor.Length][][,]; 
                for (int i = 0; i < input_tensor.Length; i++)
                {
                    h_cur[i] = new float[hidden_dim][,];
                    cur_state[i] = new float[hidden_dim][,];
                    cur_state_m[i] = new float[hidden_dim_m][,];
                    for (int j = 0; j < hidden_dim; j++)
                    {
                        h_cur[i][j] = new float[input_tensor[0][0].GetLength(0), input_tensor[0][0].GetLength(1)];
                        cur_state[i][j] = new float[input_tensor[0][0].GetLength(0), input_tensor[0][0].GetLength(1)];
                   
                    }
                    for (int j = 0; j < hidden_dim_m; j++)
                    {
                       
                        cur_state_m[i][j] = new float[input_tensor[0][0].GetLength(0), input_tensor[0][0].GetLength(1)];
                    }
                }
            }
           // dynamic h_cur = cur_state;
             dynamic c_cur = cur_state;//cur = Current input of H and C
            dynamic h_cur_m = cur_state_m;//cur = Current input of m


            dynamic c1 = Matrix.cat(input_tensor, h_cur, 1);
            // combined_h_c = torch.cat([input_tensor, h_cur], dim = 1)
            dynamic combined_h_c = conv_h_c.Forward(c1);
               // cc_i, cc_f, cc_g = torch.split(combined_h_c, self.hidden_dim, dim = 1)

             List<float[][][,]> cc_is = Matrix.chunk(combined_h_c, 3, 1);
            var cc_i = cc_is[0];var cc_f = cc_is[1];var cc_g = cc_is[2];

            var ii = sli.Forward(cc_i);
            var f = slf.Forward(cc_f);
            var g = slg.Forward(cc_g);

            var combined_m = Matrix.cat(input_tensor, h_cur_m, 1);
            // combined_m = torch.cat([input_tensor, h_cur_m], dim = 1)
            combined_m =conv_m.Forward(combined_m);
            List<float[][][,]> cc_i_m = Matrix.chunk(combined_m, 3, 1);
          
            var c_next = Matrix.MatrixAdd(mulf_1.Forward(f, c_cur), mulig_1.Forward(ii, g));

            var i_m = sli_m.Forward(cc_i_m[0]);
            var f_m = slf_m.Forward(cc_i_m[1]);
            var g_m = slg_m.Forward(cc_i_m[2]);

            var h_next_m = Matrix.MatrixAdd(mulf.Forward(f_m, h_cur_m), mulig.Forward(i_m, g_m));
           
            dynamic c2 = Matrix.cat(c_next, h_next_m, 1);
            dynamic c3 = Matrix.cat(c1, c2, 1);
            dynamic combined_o = conv_o.Forward(c3);
            var o=  slcombined_o.Forward(combined_o);
         //   var h_next =Matrix.cat(c_next, h_next_m, 1);
           var  h_next= conv_h_next.Forward(c2);
             h_next= mulo.Forward(o, h_next_tl.Forward(h_next));
            //    h_cur, c_cur = cur_state  #cur = Current input of H and C
            //h_cur_m = cur_state_m #cur = Current input of m

            //combined_h_c = torch.cat([input_tensor, h_cur], dim = 1)
            //combined_h_c = self.conv_h_c(combined_h_c)
            //cc_i, cc_f, cc_g = torch.split(combined_h_c, self.hidden_dim, dim = 1)

            //combined_m = torch.cat([input_tensor, h_cur_m], dim = 1)
            //combined_m = self.conv_m(combined_m)
            //cc_i_m, cc_f_m, cc_g_m = torch.split(combined_m, self.hidden_dim_m, dim = 1)

            //i = torch.sigmoid(cc_i)
            //f = torch.sigmoid(cc_f)
            //g = torch.tanh(cc_g)
            //c_next = f * c_cur + i * g

            //i_m = torch.sigmoid(cc_i_m)
            //f_m = torch.sigmoid(cc_f_m)
            //g_m = torch.tanh(cc_g_m)
            //h_next_m = f_m * h_cur_m + i_m * g_m

            //combined_o = torch.cat([input_tensor, h_cur, c_next, h_next_m], dim = 1)
            //combined_o = self.conv_o(combined_o)
            //o = torch.sigmoid(combined_o)

            //h_next = torch.cat([c_next, h_next_m], dim = 1)
            //h_next = self.conv_h_next(h_next)
            //h_next = o * torch.tanh(h_next)
            

            return (h_next, c_next, h_next_m);
        }
        public dynamic Backward(dynamic grid)
        {

            var b_o = mulo.Backward(grid);
            var b_h_next_tl= mulo.BackwardY(grid);
           var b_conv_h_next= h_next_tl.Backward(b_h_next_tl);
            conv_h_next.backweight(b_conv_h_next);
           float[][][,] b_h_next =  conv_h_next.Backward(b_conv_h_next);

            //  List<float[][][,]> dynamics = Matrix.chunk(b_h_next, 2, 1);
            var b_c_next = new float[grid.Length][][,];
            var b_h_next_m=new float[grid.Length][][,];
            for (int i = 0; i < grid.Length; i++)
                b_c_next[i] = b_h_next[i].Take(hidden_dim).ToArray();
            for (int i = 0; i < grid.Length; i++)
                b_h_next_m[i] = b_h_next[i].Take(hidden_dim_m).ToArray();
      
            var b_combined_o= slcombined_o.Backward(b_o);
            conv_o.backweight(b_combined_o);
            var b_c3 = conv_o.Backward(b_combined_o);
            //List<float[][][,]> dynamics2 = Matrix.chunk(b_c3, 2, 1);
            //var b_c1 = dynamics2[0];
           var b_f_m= mulf.Backward(b_h_next_m);
            var b_h_cur_m = mulf.BackwardY(b_h_next_m);
            var b_i_m = mulig.Backward(b_h_next_m);
            var b_g_m = mulig.BackwardY(b_h_next_m);

            var b_cc_i_m_2=  slg_m.Backward(b_g_m);
            var b_cc_i_m_1 = slf_m.Backward(b_f_m);
            var b_cc_i_m_0 = sli_m.Backward(b_i_m);

            var b_f = mulf_1.Backward(b_c_next);
            var b_c_cur = mulf_1.BackwardY(b_c_next);
            var b_i = mulig_1.Backward(b_c_next);
            var b_g = mulig_1.BackwardY(b_c_next);

            var b_cc_i = sli.Backward(b_i);
            var b_cc_f = slf.Backward(b_f);
            var b_cc_g = slg.Backward(b_g);

           var b_combined_m= Matrix.cat( Matrix.cat(b_cc_i_m_0, b_cc_i_m_1, 1), b_cc_i_m_2,1);
            conv_m.backweight(b_combined_m);
            b_combined_m = conv_m.Backward(b_combined_m);

            float[][][,] b_combined_h_c = Matrix.cat(Matrix.cat(b_cc_i, b_cc_f, 1), b_cc_g,1);
            conv_h_c.backweight(b_combined_h_c);
            b_combined_h_c = conv_h_c.Backward(b_combined_h_c);
            var b_next = new float[grid.Length][][,];
            for (int i = 0; i < grid.Length; i++)
                b_next[i] = b_combined_h_c[i].Take(input_dim).ToArray();  
           
            return b_next;
        }

        public void update(float lr)
        {
            conv_h_c.update(lr);
            conv_m.update(lr);
            conv_o.update(lr);
            conv_h_next.update(lr);
        }
    }
}
