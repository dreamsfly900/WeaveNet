using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{

    
    public class LeakyReLU : Layer
    {
        public float f = 0.1f;
        public dynamic Dx;
        public dynamic Backward(dynamic dout)
        {
            return Matrix.ReLubackward(Dx,dout, f);
        }

        public dynamic Forward(dynamic x )
        {
            Dx = x;
            return Matrix.ReLu(x, f);
        }

        
    }
    public class ReLuLayer : Layer
    {
        public dynamic Dx;
        public dynamic Backward(dynamic dout)
        {
            return Matrix.ReLubackward(Dx,dout, 0);
        }

        public dynamic Forward(dynamic x)
        {
            Dx = x;
            return Matrix.ReLu(x,0);
           
        }
    }
    public class TanhLayer : Layer
    {
        dynamic convoutdata;
        public dynamic Forward(dynamic x)
        {
            if (x is float[][][,])
                convoutdata = Matrix.activation_tanh(x);
            else if(x is float[][])
                convoutdata = Matrix.activation_tanh(x);
            else if (x is float[])
                convoutdata = Matrix.activation_tanh(x);
            else if (x is float)
                convoutdata = Forward(x);
            else
                convoutdata = Matrix.activation_tanh(x).values;
            return convoutdata;
        }
        public dynamic Backward(dynamic dout)
        {
            if (dout is float[][][,])
                return Matrix.activation_tanhbackward(convoutdata, dout);
            else if(dout is float[][])
                return Matrix.activation_tanhbackward(convoutdata, dout);
            else if (dout is float[])
                return Matrix.activation_tanhbackward(convoutdata, dout);
            else if (dout is float)
                return backward(convoutdata, dout);
            else
                return Matrix.activation_tanhbackward(convoutdata, dout).values;

           
        }
      
        float backward(float outdata, float dout)
        {
            float dx = dout * (1.0f - (float)(Math.Pow(outdata, 2)));

            return dx;
        }
        
        float outdata;
         float Forward(float x)
        {
            outdata = (float)Math.Tanh(x);
            return outdata;
        }
         float Backward(float dout)
        {
            float dx = dout * (1.0f - (float)(Math.Pow(outdata, 2)));

            return dx;
        }
    }
    public class SigmodLayer : Layer
    {
        dynamic convoutdata;
        public dynamic Forward(dynamic x)
        {
            if(x is float[][][,])
                convoutdata = Matrix.activation_Sigma(x);
            else if (x is float[])
                convoutdata = Matrix.activation_Sigma(x);
            else if(x is float[][])
                convoutdata = Matrix.activation_Sigma(x);
            else if (x is float)
                convoutdata = forward(x);
            else
                convoutdata = Matrix.activation_Sigma(x).values;
            return convoutdata;
        }
        public dynamic Backward(dynamic dout)
        {

            if(dout is float[][][,])
                return Matrix.activation_Sigmabackward(convoutdata, dout);
            else if(dout is float[])
                return Matrix.activation_Sigmabackward(convoutdata, dout);
            else if(dout is float[][])
                return Matrix.activation_Sigmabackward(convoutdata, dout);
            else if (dout is float)
                return backward( dout);
            else
               return Matrix.activation_Sigmabackward(convoutdata, dout).values;

             
        }

       
        float outdata;
         float forward(float x)
        {
            outdata = (float)(1.0 / (1.0 + Math.Exp(-x)));
            return outdata;
        }
         float sigmoid_der(float z)
        {
            return forward(z) * (1 - forward(z));
        }
         float backward(float dout)
        {
            float dx = dout * (1 - outdata) * outdata;

            return dx;
        }
    }
}
