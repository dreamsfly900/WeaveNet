using DenseCRF;
using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{
   public class ConvLayer
    {
        public int inChannels;   //输入图像的数目
        public int outChannels;  //输出图像的数目

        public dynamic grid;
        public int w;
        public int h;
        public int Kszie;
        public float[][] weights;

        public float[] basicData;
        public ConvLayer(int innum = 1, int outnum = 6, bool initW = true)
        {

            inChannels = innum;
            outChannels = outnum;
            if (initW)
                weights = util.initweights(innum, outnum);

            basicData = new float[outnum];
            if (initW)
            {
                Random rand = new Random();
                for (var b = 0; b < outnum; b++)
                    basicData[b] = ((float)rand.Next(Int32.MinValue, Int32.MaxValue) / Int32.MaxValue) * 0.1f;
            }
        }
        dynamic inputDatamatrices;
        public dynamic Forward(dynamic matrices)
        {
            //if(matrices is float[,])
            //    matrices 
            inputDatamatrices = matrices;
           var inputDatamatrices2 = conv(matrices);
            return inputDatamatrices2;
        }
        public void update(float lr = 0.1f)
        {


            weights = Matrix.MatrixSub(weights, Matrix.multiply(gridK.grid, lr));
            basicData = Matrix.MatrixSub(basicData, Matrix.multiply(gridK.basic, lr));



        }
        private dynamic conv(float[][] matrices)
        {
            float[][] data = new float[matrices.Length][];//outChannels
            data = Matrix.dot(matrices, weights);
            for (var i = 0; i < (matrices.Length); i++)
            {
                
                for (var j = 0; j < (outChannels); j++)
                    data[i][j] += basicData[j];
            }
            return data;
        }

        public dynamic backweight(dynamic grid)
        {
             
            if (grid is float[][])
            {
                //  var   weightst = Matrix.T(weights);
                //var ss = Matrix.dot(grid, weightst);
                var gridss = Matrix.T(grid);
                var ss= Matrix.dot(gridss, inputDatamatrices );
              
                float[] outputB = new float[outChannels];

                for (var s = 0; s < outChannels; s++)
                {
                    

                    float sum = Matrix.sum(gridss[s]);

                    outputB[s] = sum / outChannels;
                    //outputB[s] = sum;
                }

                 var gridd = new { grid = Matrix.T(ss), basic = outputB };
                gridK = gridd;
                return gridd;
                   
               
            }
            return grid;
        }
        dynamic gridK;
        public dynamic backward(dynamic grid)
        {

            if (grid is float[][])
            {
               var   weightst = Matrix.T(weights);
                var ss = Matrix.dot(grid, weightst);
               
                return ss;


            }
            return grid;
        }
    }
}
