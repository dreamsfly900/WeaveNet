using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Layer
{
  public  class Averpooling
    {
		int stride = 2;
		public Averpooling(int _stride=2)
		{
			stride = _stride;
		}
		public dynamic Forward(dynamic matrices)
        {
			return pooing(matrices);

		}

		float[][][,]  pooing(float[][][,] data)
		{
			int i = 0;int count= data.Length;

			float[][][,] data2 = new float[count][][,];
			for (i = 0; i < count; i++)
			{
				int count2 = data[i].Length;
				data2[i] = new float[count2][,];
				for (int j = 0; j < count2; j++)
				{
					//if (data2[i][j] == null)
					//	data2[i][j] = new float[data[i][j].GetLength(0), data[i][j].GetLength(1)];
					data2[i][j] = Matrix.averPooling(data[i][j], stride);
				}  
			}

			return data2;
		}
		public dynamic backward(dynamic grid)
        {
			return backpooling(grid);
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
					data2[i][j] = Matrix.kroneckerAvg(data[i][j], stride);
				}
			}
			return data2; 

		}
    }
}
