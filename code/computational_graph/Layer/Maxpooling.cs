using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{ 

    public class Maxpooling
    {
        public dynamic Forward(dynamic matrices)
        {
           return MAXpooling(matrices);
        }
        int stride = 2;
        public Maxpooling(int _stride = 2)
        {
            stride = _stride;
        }
        dynamic Ddata;
        float[][][,] MAXpooling(float[][][,] data)
        {
            int i = 0; int count = data.Length;

            float[][][,] data2 = new float[count][][,];
            Matrix[][] data2m = new Matrix[count][];
            for (i = 0; i < count; i++)
            {
                int count2 = data[i].Length;
                data2[i] = new float[count2][,];
                data2m[i]= new Matrix[count2];
                for (int j = 0; j < count2; j++)
                {
                    //if (data2[i][j] == null)
                    //	data2[i][j] = new float[data[i][j].GetLength(0), data[i][j].GetLength(1)];
                    //Matrix.MaxPooling(recla.matrixs[ha, hi], cla.stride);
                    data2m[i][j] = Matrix.MaxPooling(data[i][j], stride);
                    data2[i][j] = data2m[i][j].values;
                }
            }
            Ddata = data2m;
            return data2;
            
        }
        float[][][,] backpooling(float[][][,] data)
        {
            int i = 0; int count = data.Length;

            float[][][,] data2 = new float[count][][,];
            for (i = 0; i < count; i++)
            {
                int count2 = data[i].Length;
                data2[i] = new float[count2][,];
                for (int j = 0; j < count2; j++)
                {
                    //if (data2[i][j] == null)
                    //	data2[i][j] = new float[data[i][j].GetLength(0), data[i][j].GetLength(1)];
                    data2[i][j] = Matrix.kroneckerMax(data[i][j], stride, Ddata[i][j].poolxies);
                }
            }
            return data2;

        }
        public dynamic backward(dynamic grid)
        {
           return backpooling(grid);
        }
    }
}
