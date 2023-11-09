
using computational_graph;
using computational_graph.Layer;
using computational_graph.loss;
using computational_graph.Util;
using DenseCRF;
using System;

namespace WeaveNetE
{
    internal class Program
    {
        static void Main(string[] args)
        {
            System.AppContext.SetSwitch("System.Drawing.EnableUnixSupport", true);
            String path = AppDomain.CurrentDomain.BaseDirectory;
            Console.WriteLine("WeaveNet深度学习计算架构");
            if (args.Length > 0)
            {
                if (args[0] == "V")
                {
                    Console.WriteLine("1.0.0");
                }
                if (args[0] == "pearson")
                { 
                    
                    float[] data = new float[] { 1, 2, 3, 4, 5 };
                    float[] data2 = new float[] { 2 ,2, 1, 4, 3 };
                    Console.WriteLine("参数1： 1, 2, 3, 4, 5");
                    Console.WriteLine("参数2： 2 ,2, 1, 4, 3 ");
                    var sim = Sim.sim_pearson(data, data2);
                    Console.WriteLine("皮尔逊相关度测试:" + sim);
                    
                }
                if (args[0] == "MSE")
                {
                    MSELoss mSELoss = new MSELoss();
                    // 1000
                    var hh = mSELoss.Forward(new float[] { 2f }, new float[] { 5f });
                    Console.WriteLine("参数1： 2");
                    Console.WriteLine("参数2： 5 ");
                    Console.WriteLine("MES:" + hh);
                }
                if (args[0] == "bp")
                {
                    Console.WriteLine("bp网络测试： ");
                    BPTest.BP();
                }
                if (args[0] == "Forward")
                {
                    Console.WriteLine("数值2正传播...");
                    ConvLayer cl1 = new ConvLayer(1, 5, true);
                    float[][] xx2 = new float[1][];
                        xx2[0] = new float[] { 2 };
                    var ff = cl1.Forward(xx2);
                    util.prirt(ff);
                }
                if (args[0] == "Sigmod")
                {
                    Console.WriteLine("激活函数Sigmod正传播");
                    SigmodLayer sl = new SigmodLayer();
                    float[][] xx2 = new float[1][];
                    xx2[0] = new float[] { 2 };
                    var ff=sl.Forward(xx2);
                    util.prirt(ff);
                }
                if (args[0] == "backward")
                {
                    Console.WriteLine("数值2反传播");
                    ConvLayer cl1 = new ConvLayer(1, 5, true);
                    float[][] xx2 = new float[1][];
                    xx2[0] = new float[] { 2 };
                    var ff = cl1.Forward(xx2);
                    var ff2=cl1.backward(ff);
                    util.prirt(ff2);
                }
                if (args[0] == "BCELoss")
                {
                    BCELoss cELoss = new BCELoss();
                    var hh = cELoss.Forward(new float[] { 0.2f }, new float[] { 0.5f });
                    Console.WriteLine("参数1： 0.2");
                    Console.WriteLine("参数2： 0.5 ");
                    Console.WriteLine("BCELoss:" + hh);
                  
                }
                if (args[0] == "cross")
                {

                    cross_entropy cELoss = new cross_entropy();
                    var hh = cELoss.Forward(new float[] { 2f }, new float[] { 5f });
                    Console.WriteLine("参数1： 2");
                    Console.WriteLine("参数2： 5 ");
                    Console.WriteLine("BCELoss:" + hh);
                }
                if (args[0] == "Softmax")
                {
                    Softmax cELoss = new Softmax();
                    var hh = cELoss.Forward(new float[] { 0.2f, 0.5f });
                    Console.WriteLine("参数1： 2,5");
                   
                    Console.WriteLine("Softmax:" );
                    util.prirt(hh);
                }
            }
        }
    }
}