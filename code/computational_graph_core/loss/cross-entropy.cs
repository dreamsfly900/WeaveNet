using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.loss
{
  public  class cross_entropy
    {
       
        public dynamic Backward()
        {
            return Backward(dx, dy);
        }
        dynamic Backward(float[] DX, float[] DY)
        {
            //self.dnx = -self.ny / self.nx
            float[] gird = new float[DY.Length];
            for (int i = 0; i < DY.Length; i++)
            {
                gird[i] = -DY[i]/DX[i];
            }
            return gird;
        }
        dynamic Backward(float[][] DX, float[][] DY)
        {
            float[][] gird = new float[DY.Length][];
            for (int i = 0; i < DY.Length; i++)
            {
                gird[i] = new float[DY[i].Length];
                for (int j = 0; j < DY[i].Length; j++)
                {
                    gird[i][j] = -DY[i][j]/DX[i][j];
                }

            }

            //loss = np.sum(-ny * np.log(nx))
            return gird;
        }
        dynamic Backward(float[][][,] DX, float[][][,] DY)
        {
            float[][][,] gird = new float[DY.Length][][,];
            for (int i = 0; i < DY.Length; i++)
            {
                gird[i] = new float[DY[i].Length][,];
                for (int j = 0; j < DY[i].Length; j++)
                {
                    gird[i][j] = Backward(DY[i][j] , DX[i][j]);
                }

            }

            //loss = np.sum(-ny * np.log(nx))
            return gird;
        }
        dynamic Backward(float[,] DX, float[,] DY)
        {
            float[,] gird = new float[DY.GetLength(0), DY.GetLength(1)];
            for (int i = 0; i < DY.GetLength(0); i++)
            {
                
                for (int j = 0; j < DY.GetLength(1); j++)
                {
                    gird[i,j] = -DY[i,j] / DX[i,j];
                }

            }

            //loss = np.sum(-ny * np.log(nx))
            return gird;
        }
        dynamic dx, dy;
        public float Forward(dynamic DX, dynamic DY)
        {
            dx = DX;
            dy = DY;
            return entropy(DX, DY);
        }
        dynamic entropy(float[] DX, float[] DY)
        {
            float loss = 0;
            for (int i = 0; i < DY.Length; i++)
            {
                loss += -DY[i] * (float)Math.Log(DX[i]);
            }

            //loss = np.sum(-ny * np.log(nx))
            return loss;
        }
        dynamic entropy(float[][] DX, float[][] DY)
        {
            float loss = 0;
            for (int i = 0; i < DY.Length; i++)
            {
                for (int j = 0; j < DY[i].Length; j++)
                {
                    loss += -DY[i][j] * (float)Math.Log(DX[i][j]);
                }
               
            }

            //loss = np.sum(-ny * np.log(nx))
            return loss;
        }
        dynamic entropy(float[][][,] DX, float[][][,] DY)
        {
            float loss = 0;
            for (int i = 0; i < DY.Length; i++)
            {
                for (int j = 0; j < DY[i].Length; j++)
                {
                    loss += entropy(DY[i][j] , DX[i][j]);
                }

            }

            //loss = np.sum(-ny * np.log(nx))
            return loss;
        }
        dynamic entropy(float[,] DX, float[,] DY)
        {
            float loss = 0;
            for (int i = 0; i < DY.GetLength(0); i++)
            {
                for (int j = 0; j < DY.GetLength(1); j++)
                {
                    loss += -DY[i,j] * (float)Math.Log(DX[i,j]);
                }

            }

            //loss = np.sum(-ny * np.log(nx))
            return loss;
        }
    }
}
