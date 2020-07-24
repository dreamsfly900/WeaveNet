using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.loss
{
    public class BCELoss
    {
        public dynamic Backward(dynamic dout)
        {
            return null;
        }
        dynamic dx, dy;
        public float Forward(dynamic DX, dynamic DY)
        {
            dx = DX;
            dy = DY;
            return BCE(DX, DY);
        }
         
        public float BCE(float[] Data, float[] Data2)
        {
            float fSum = 0;
            int Num = Data.GetLength(0);
            for (int i = 0; i < Num; ++i)
            {
                fSum += Data[i] * (float)Math.Log(Data2[i]) + (1 - Data[i]) * (float)Math.Log((1 - Data2[i]));

            }
            return (float)(-(fSum)) / (float)Num;
        }
        public float BCE(float[,] Data, float[,] Data2)
        {
          
            int X = Data.GetLength(0);
            int y = Data.GetLength(1);
           
            float Sum = 0;
            for (int i = 0; i < X; ++i)
            {
                float fSum = 0;
                for (int j = 0; j < y; ++j)
                    fSum +=( Data[i, j] * (float)Math.Log(Data2[i, j]) + (1 - Data[i, j]) * (float)Math.Log((1 - Data2[i, j])));
                fSum = -(fSum / y);
                Sum += fSum;
            }
            return (float)(Sum) / (float)(X);
        }
    }
}
