using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Util
{
   public class Sim
    {
        /// <summary>
        /// 皮尔逊相关度
        /// </summary>
        /// <param name="data1"></param>
        /// <param name="data2"></param>
        /// <returns></returns>
        public static float sim_pearson(float[] data1,float[] data2)
        {
            if (data1.Length != data2.Length)
                return 0;
            float sum =0, sum1_sq=0;
            float sum2 = 0, sum2_sq=0, asum=0;
            for (int i = 0; i < data1.Length; i++)
            {
                sum += data1[i];
                sum2 += data2[i];
                sum1_sq += (float)Math.Pow(data1[i], 2.0);
                sum2_sq += (float)Math.Pow(data2[i], 2.0);
                asum += data1[i] * data2[i];
            }
            var n = data1.Length;
            var num = asum - (sum * sum2 / n);

            var den = Math.Sqrt((sum1_sq - Math.Pow(sum, 2) / n) * (sum2_sq - Math.Pow(sum2, 2) / n));

            if (den == 0)
                return 0;

            return (float)(num / den);

        }
    }
}
