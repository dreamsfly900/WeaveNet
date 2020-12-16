using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.loss
{
    public class MSELoss
    {
        dynamic dx, dy;
        public float Forward(dynamic DX, dynamic DY)
        {
           var lost= MSE(DX, DY);
           // if (dx == null)
            {
                dx = DX;
                dy = DY;
            }
            //else
            //{
            //    dx = Matrix.MatrixAdd(dx, DX);
            //    dy = Matrix.MatrixAdd(dy, DY);
            //}
            return lost;
        }
        public dynamic Backward()
        {


            return backward(dx, dy);
        }
        dynamic backward(float[] Data, float[] Data2)
        {
            int Num = Data.GetLength(0);
            float[] grad = new float[Num];
            for (int i = 0; i < Num; ++i)
            {
                //dx = 2 * (self.x - self.y) / self.x.size
                grad[i] = 2 * (Data[i] - Data2[i]) / Num;
            }
            return grad;
        }
        dynamic backward(float[,] Data, float[,] Data2)
        {
            int Num = Data.GetLength(0);
            int y = Data.GetLength(1);
            float[,] grad = new float[Num, y];
            for (int i = 0; i < Num; ++i)
            {
                //dx = 2 * (self.x - self.y) / self.x.size
                for (int j = 0; j < y; ++j)
                    grad[i, j] = 2 * (Data[i, j] - Data2[i, j]) / (Num * y);
            }

            return grad;
        }
        dynamic backward(float[][] Data, float[][] Data2)
        {
            int Num = Data.GetLength(0);
            int y = Data[0].GetLength(0);
            float[][] grad = new float[Num][];
            for (int i = 0; i < Num; ++i)
            {
                grad[i] = new float[y];
                //dx = 2 * (self.x - self.y) / self.x.size
                for (int j = 0; j < y; ++j)
                    grad[i][ j] = 2 * (Data[i][ j] - Data2[i][ j]) / (Num * y);
            }

            return grad;
        }
        dynamic backward(float[][][,] Data, float[][][,] Data2)
        {
            int s = Data.GetLength(0);
            int ss = Data[0].GetLength(0);
            int Num = Data[0][0].GetLength(0);
            int y = Data[0][0].GetLength(1);
            float[][][,] grad = new float[s][][, ];

            for (int si = 0; si < s; ++si)
            {
                grad[si] = new float[ss][,];
                for (int sy = 0; sy < ss; ++sy)
                {
                    grad[si][sy] = new float[Num, y];
                    for (int i = 0; i < Num; ++i)
                    {
                        //dx = 2 * (self.x - self.y) / self.x.size
                        for (int j = 0; j < y; ++j)
                            grad[si][sy][i, j] = 2 * (Data[si][sy][i, j] - Data2[si][sy][i, j]) / (Num * y);
                    }
                }
            }

            return grad;
        }
        float MSE(float[] Data, float[] Data2)
        {
            float fSum = 0;
            int Num = Data.GetLength(0);
            for (int i = 0; i < Num; ++i)
            {
                fSum += (Data[i] - Data2[i]) * (Data[i] - Data2[i]);
            }
            return (float)(fSum / (float)Num);
        }
        // np.sum(np.square(x - y)) / x.size
        float MSE(float[,] Data, float[,] Data2)
        {
            float fSum = 0;
            int X = Data.GetLength(0);
            int y = Data.GetLength(1);
            for (int i = 0; i < X; ++i)
            {
                for (int j = 0; j < y; ++j)
                    fSum += (Data[i, j] - Data2[i, j]) * (Data[i, j] - Data2[i, j]);
            }
            return (float)(fSum / (float)(X * y));
        }
        float MSE(float[][] Data, float[][] Data2)
        {
            float fSum = 0;
            int X = Data.GetLength(0);
            int y= Data[0].GetLength(0);
            for (int i = 0; i < X; ++i)
            {
               
                for (int j = 0; j < y; ++j)
                    fSum += (Data[i][ j] - Data2[i][ j]) * (Data[i][ j] - Data2[i][j]);
            }
            return (float)(fSum / (float)(X * y));
        }
        float MSE(float[][][,] Data, float[][][,] Data2)
        {
            float fSum = 0;
            int X = Data.GetLength(0);
            int y = Data[0].GetLength(0);
            int S = Data[0][0].GetLength(0);
            int S2 = Data[0][0].GetLength(1);
            for (int i = 0; i < X; ++i)
            {
                for (int j = 0; j < y; ++j)
                    for (int s = 0; s < S; ++s)
                        for (int s2 = 0; s2 < S2; ++s2)
                            fSum += (Data[i][ j][ s, s2] - Data2[i][ j][ s, s2]) * (Data[i][ j][ s, s2] - Data2[i][ j][ s, s2]);
            }
            return (float)(fSum / (float)(X * y* S* S2));
        }

        internal float[] MESBasic(dynamic dW, int outnum)
        {
            float[] outputB = new float[outnum];
            for (var s = 0; s < outnum; s++)
            {
                float sum = 0f;
                for (var ss = 0; ss < dW.GetLength(0); ss++)
                {
                    sum += Sum(dW[ss, s]);
                }
                outputB[s] = sum;
            }
            return outputB;
        }
        float Sum(Matrix input)
        {
            float sum = 0f;
            int m = input.values.GetLength(0);
            int n = input.values.GetLength(1);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {

                    sum += input.values[0, 0];

                }
            }
            return sum;
        }
    }



}