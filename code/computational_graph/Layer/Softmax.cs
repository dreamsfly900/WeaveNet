using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{
    public class Softmax : Layer
    {
        

    //    def stable_softmax(x):
    //exps = np.exp(x-np.max(x))
    //return exps / np.sum(exps)
        public dynamic Backward(dynamic dout)
        {
            return Backward(dout);
        }
        // def softmax(x):
        // exps = np.exp(x)
        // return exps / np.sum(exps)
        dynamic dx;
        public dynamic Forward(dynamic x)
        {

            dx = Forward(x);
          return dx;
        }
        dynamic Backward(float[] x )
        {
            return Backward(x, dx);
        }
      dynamic Backward(float[] x, float[] dx)
        {
            
            
            int len = x.Length;
          
            float[] grad = new float[x.Length];
            for (int i = 0; i < len; i++)
            {
                float[] temp=  Matrix.multiply(dx, dx[i]);
                temp[i] -= dx[i];
                for (int j = 0; j < len; j++)
                {
                    temp[j] = -temp[j];
                }
                temp = Matrix.multiply(temp, x);
                grad[i]= Matrix.sum(temp);
            }
           
            return grad;
        }
        dynamic Backward(float[][] x)
        {
            int len = x.Length;
            dynamic grad = new float[len][];
           
            for (int i = 0; i < len; i++)
            {
                int lenj = x[i].Length;
                grad[i] = new float[lenj];
                grad[i] = Backward(x[i], dx[i]);
             
            }
            
            return grad;
        }
        dynamic Backward(float[][][,] x)
        {
            int len = x.Length;
            dynamic exps = new float[len][][,];

            for (int i = 0; i < len; i++)
            {
                int lenj = x[i].Length;
                exps[i] = new float[lenj][,];

                for (int j = 0; j < lenj; j++)
                {
                    exps[i][j] = Forward(x[i][j]);
                }
            }

            return exps;
        }
        dynamic Backward(float[,] x)
        {
            int len = x.GetLength(0);
            int lenj = x.GetLength(1);
            dynamic exps = new float[len, lenj];
            
            
            for (int i = 0; i < len; i++)
            {
                float[] temp = new float[lenj];
                float[] tempdx = new float[lenj];
                for (int j = 0; j < lenj; j++)
                {
                    temp[j] = x[i, j];
                    tempdx[j] = dx[i, j];
                }
               float[] aa= Backward(temp, tempdx);
                for (int j = 0; j < lenj; j++)
                    exps[i, j] = aa[j];

            }
            
            return exps;
        }


        dynamic Forward(float[] x)
        {
            dynamic exps = new float[x.Length];
            float sum = 0f;
            int len = x.Length;
            float max = Matrix.Max(x);
            for (int i = 0; i < len; i++)
            {
                 exps[i] = (float)Math.Exp(x[i]- max);
                sum += exps[i];
            }
            for (int i = 0; i < len; i++)
            {
                exps[i] = exps[i] / sum;
            }
            return exps ; 
        }
        dynamic Forward(float[][] x)
        {
            int len = x.Length;
            dynamic exps = new float[len][];
           
           
            for (int i = 0; i < len; i++)
            {
                
                int lenj = x[i].Length;
                exps[i] = new float[lenj];
                exps[i]=  Forward(x[i]);
                
            }
            
            return exps;
        }
        dynamic Forward(float[][][,] x)
        {
            int len = x.Length;
            dynamic exps = new float[len][][,];
            
            for (int i = 0; i < len; i++)
            {
                int lenj = x[i].Length;
                exps[i] = new float[lenj][,];

                for (int j = 0; j < lenj; j++)
                {
                    exps[i][j]= Forward(x[i][j]);
                }
            }
             
            return exps;
        }
        dynamic Forward(float[,] x)
        {
            int len = x.GetLength(0);
            int lenj = x.GetLength(1);
            dynamic exps = new float[len, lenj];
            
            for (int i = 0; i < len; i++)
            {
                float[] temp = new float[lenj];
                for (int j = 0; j < lenj; j++)
                {
                    temp[j] = x[i, j];
                }
                float[] aa = Forward(temp);
                for (int j = 0; j < lenj; j++)
                    exps[i, j] = aa[j];
            }
             
            return exps;
        }
    }
}
