using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{
    public class Dropout : Layer
    {
        public float dropout_ratio = 0.5f;
        dynamic mask;
        public dynamic Forward(dynamic dout)
        {
             mask=Matrix.dropout( dout, dropout_ratio);
            return mask;
        }
 
        public dynamic Backward(dynamic x)
        {
            return Matrix.multiply(x, mask);
        }


    }
}
