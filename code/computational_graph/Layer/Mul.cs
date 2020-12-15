using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{
    public class MulLayer
    {
        dynamic xx, yy;
        public dynamic Forward(dynamic x, dynamic y)
        {
            xx = x;
            yy = y;
            return Matrix.multiply(x, y);

        }

       
        public dynamic Backward(dynamic dout)
        {
            dynamic dx;
           
                dx = Matrix.multiply(dout, yy);
            return dx;
        }
       
       
        public dynamic BackwardY(dynamic dout)
        {
            dynamic dx;
            
                dx = Matrix.multiply(dout, xx);
             
            return dx;
        }

        float x = 0, y = 0;
        public float Forward(float _x, float _y)
        {

            x = _x;
            y = _y;
            return x * y;
        }
        public float Backward(float dout)
        {
            float dx = dout * y;

            return dx;
        }
    }
}
