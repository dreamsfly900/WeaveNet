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
        int hidden_dim; int hidden_dim_m;
        public PredRNN_Cell(int input_size,int  input_dim,int  hidden_dim_m,int  hidden_dim,int kernel_size,bool bias)
        {
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
        SigmodLayer sli_m = new SigmodLayer();
        SigmodLayer slf_m = new SigmodLayer();
        TanhLayer slg_m = new TanhLayer();
        SigmodLayer slcombined_o = new SigmodLayer();
        public dynamic Forward(dynamic input_tensor, dynamic cur_state, dynamic cur_state_m)
        {
            dynamic h_cur= cur_state, c_cur = cur_state;//cur = Current input of H and C
            dynamic h_cur_m = cur_state_m;//cur = Current input of m


            dynamic c1 = Matrix.cat(input_tensor, h_cur, 1);
            // combined_h_c = torch.cat([input_tensor, h_cur], dim = 1)
            dynamic combined_h_c = conv_h_c.Forward(c1);
               // cc_i, cc_f, cc_g = torch.split(combined_h_c, self.hidden_dim, dim = 1)

             List<float[][]> cc_is = Matrix.chunk(combined_h_c, hidden_dim, 1);
            var cc_i = cc_is[0];var cc_f = cc_is[1];var cc_g = cc_is[2];
            var combined_m = Matrix.cat(input_tensor, h_cur, 1);
            // combined_m = torch.cat([input_tensor, h_cur_m], dim = 1)
            combined_m =conv_m.Forward(combined_m);
            List<float[][]> cc_i_m = Matrix.chunk(combined_m, hidden_dim_m, 1);
            var i = sli.Forward(cc_i);
            var f = sli.Forward(cc_f);
            var g = sli.Forward(cc_g);
            var c_next = Matrix.MatrixAdd(mulf.Forward(f, c_cur), mulig.Forward(i,g));

            var i_m = sli_m.Forward(cc_i_m[0]);
            var f_m = slf_m.Forward(cc_i_m[1]);
            var g_m = slg_m.Forward(cc_i_m[2]);

            var h_next_m = Matrix.MatrixAdd(mulf.Forward(f_m, h_cur_m), mulig.Forward(i_m, g_m));
           
            dynamic c2 = Matrix.cat(c_next, h_next_m, 1);
            dynamic c3 = Matrix.cat(c1, c2, 1);
            dynamic combined_o = conv_o.Forward(c3);
            var o=  slcombined_o.Forward(combined_o);
            //combined_m = self.conv_m(combined_m)
            // cc_i_m, cc_f_m, cc_g_m = torch.split(combined_m, self.hidden_dim_m, dim = 1)
            //    i = torch.sigmoid(cc_i)
            //f = torch.sigmoid(cc_f)
            //g = torch.tanh(cc_g)
            //c_next = f * c_cur + i * g
            //    i_m = torch.sigmoid(cc_i_m)
            //f_m = torch.sigmoid(cc_f_m)
            //g_m = torch.tanh(cc_g_m)
            //h_next_m = f_m * h_cur_m + i_m * g_m



            //combined_o = torch.cat([input_tensor, h_cur, c_next, h_next_m], dim = 1)
            //combined_o = self.conv_o(combined_o)
            //o = torch.sigmoid(combined_o)


        }
    }
}
