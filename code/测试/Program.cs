using computational_graph.Layer;
using computational_graph.Util;

namespace 测试
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var a = new float[] { 10, 11, 12, 12, 12 };
            var b = new float[] { 15, 16, 17, 16, 17 };
          var tcc=  Sim.Temporal_Correlation_Coefficient(a, b);

          var nse=  Sim.NSE(a,b);
             ConvLayer cl2 = new ConvLayer(4, outnum: 5, true,true);
            var x_data = new float[3][];
            x_data[0] = new float[] { 0f, 1f, 2f,1f };
            x_data[1] = new float[] { 3f, 4f, 5f,1f };
            x_data[2] = new float[] { 6f, 7f,8f,1f };
            var grid=cl2.Forward(x_data);
            cl2.backweight(grid);
             cl2.backward(grid);
        }
    }
}
