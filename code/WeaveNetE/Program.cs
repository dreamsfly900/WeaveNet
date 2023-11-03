using computational_graph;
using computational_graph.loss;
using computational_graph.Util;
using System;

namespace WeaveNetE
{
    internal class Program
    {
        static void Main(string[] args)
        {
            System.AppContext.SetSwitch("System.Drawing.EnableUnixSupport", true);
            String path = AppDomain.CurrentDomain.BaseDirectory;
            Console.WriteLine("WeaveNet神经网络计算图");
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
            }
        }
    }
}