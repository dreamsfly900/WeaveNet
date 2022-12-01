using computational_graph.Layer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
   public class FCN
    {
        Conv2DLayer cl; 
        Conv2DLayer cl2;
        Conv2DLayer cl3;
        ConvTranspose2DLayer Tcl1;
        Maxpooling mpl = new Maxpooling();
        Maxpooling mpl2 = new Maxpooling();
        SigmodLayer sl = new SigmodLayer();
        SigmodLayer sl2 = new SigmodLayer();
        SigmodLayer sl3 = new SigmodLayer();
        Softmax sl4 = new Softmax();
        public FCN(int weightssize)
        {
            cl = new Conv2DLayer(1, weightssize / 2, weightssize, 1, 6, bias: false);
            cl2 = new Conv2DLayer(1, weightssize / 2, weightssize, 6, 12, bias: false);
            cl3 = new Conv2DLayer(1, weightssize / 2, weightssize, 12, 24, bias: false);
            Tcl1 = new ConvTranspose2DLayer(2, 1, weightssize + 1, 24, 1, bias: false);
        }
        public dynamic Forward(dynamic data)
        {
            dynamic data2= cl.Forward(data);
            data2=sl.Forward(data2);
            data2=mpl.Forward(data2);
            data2 = cl2.Forward(data2);
            data2 = sl2.Forward(data2);
            data2 = mpl2.Forward(data2);
            data2 = cl3.Forward(data2);
            data2 = sl3.Forward(data2);
            data2=Tcl1.Forward(data2);
            data2 = sl4.Forward(data2);
            return data2;
        }
        public dynamic backward(dynamic grid)
        {
            var grid2 = sl4.Backward(grid);
            grid2= Tcl1.Backward(grid2);
            grid2 = sl3.Backward(grid2);
            grid2 = cl3.Backward(grid2);
            grid2 = mpl2.Backward(grid2);
            grid2 = sl2.Backward(grid2);
            grid2 = cl2.Backward(grid2);
            grid2 = mpl.Backward(grid2);
            grid2 = sl.Backward(grid2);
            grid2 = cl.Backward(grid2);
            return grid2;
        }
    }
}
