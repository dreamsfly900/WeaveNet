using ManagedCuda;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace FCN
{
    public class poolxy
    {
        public int x,y,nx,ny;
        public float value;
    }
   public   class Matrix
    {
     
        public  static CudaContext ctx;
        public static Stream stream;
        public static CudaKernel CUDAConvKernel;
        public static CudaKernel CUDA3dConvKernel;
        public static bool cuda=false;
        public  static int O_TILE_WIDTH = 16;
        public static bool CUDA {
            set
            { 
                if (value)
                {
                    if (stream == null)
                    {
                        ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId());
                        string[] liste = Assembly.GetExecutingAssembly().GetManifestResourceNames();
                        stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(liste[0]);
                        CUDAConvKernel = ctx.LoadKernelPTX(stream, "convolution_2D_shared");
                        CUDA3dConvKernel = Matrix.ctx.LoadKernelPTX(Matrix.stream, "convolution_3D_shared");
                    }
                  
                }
                cuda = value;
            }
            get { return cuda; }
        }
       public float[,] values = new float[0, 0];
        public static float[] ReLu(float[] input, float Alpha)
        {

            float[] temp = new float[input.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {


                temp[x] = ReLu(input[x], Alpha);

            }
            return temp;
        }
        public static float[] float3DTofloat1D(float[][,] bvalue, float[] data = null)
        {
            var x = bvalue[0].GetLength(0);
            var y = bvalue[0].GetLength(1);
            if(data == null)
              data = new float[bvalue.GetLength(0)* x * y];
            for (var c = 0; c < bvalue.Length; c++) {
                for (var i = 0; i < x; i++)
                    for (var j = 0; j < y; j++)
                    {
                        data[(c*x*y)+(i * x + j)] = bvalue[c][i, j];
                    }
            }
            return data;
        }
        public static float[] float4DTofloat1D(float[][][,] bvalue, float[] data = null)
        {
            var x = bvalue[0][0].GetLength(0);
            var y = bvalue[0][0].GetLength(1);
            if (data == null)
               data = new float[((bvalue.Length* bvalue[0].Length) * x * y)];
            int ss = bvalue[0].Length * x * y;
            for (var c = 0; c < bvalue.Length; c++)
            {
                for (var c1 = 0; c1 < bvalue[c].Length; c1++)
                {
                    for (var i = 0; i < x; i++)
                        for (var j = 0; j < y; j++)
                        {
                            data[(c *ss)+ (c1 * x * y) + (i * x + j)] = bvalue[c][c1][i, j];
                        }
                }
            }
            return data;
        }

        public static float[][][,] Pow(float[][][,] mu1, int v)
        {
            float[][][,] temp = new float[mu1.GetLength(0)][][,];
            for (var x = 0; x < mu1.GetLength(0); x++)
            {
                temp[x] = new float[mu1[x].GetLength(0)][,];
                for (var y = 0; y < mu1[x].GetLength(0); y++)
                {
                    temp[x][y] = Pow(mu1[x][y], v);
                }

            }
            return temp;
        }

        public static float Mean(float[,] ssim_map)
        {
            float sumnum = 0;
            int m = ssim_map.GetLength(0);
            int n = ssim_map.GetLength(1);
            for (var x = 0; x < m; x++)
            {

                for (var y = 0; y < n; y++)
                {
                    sumnum+=ssim_map[x, y] ;
                }

            }
            return sumnum / (m * n);
        }

        public static float[,] Pow(float[,] v1, int v2)
        {
            float[,] mu1 = new float[v1.GetLength(0), v1.GetLength(1)];
            for (var x = 0; x < v1.GetLength(0); x++)
            {
               
                for (var y = 0; y < v1.GetLength(1); y++)
                {
                    mu1[x, y] =(float)Math.Pow((double)v1[x,y], (double)v2);
                }

            }
            return mu1;
        }

        public static float[] float2DTofloat1D(float[,] bvalue)
        {

            var x = bvalue.GetLength(0);
            var y = bvalue.GetLength(1);
            float[] data = new float[x * y];
            
            for (var i = 0; i < x; i++)
                for (var j = 0; j < y; j++)
                {
                    data[i * x + j]= bvalue[i,j];
                }
            return data;
        }

        public static float[][] ReLu(float[][] input, float Alpha)
        {

            float[][] temp = new float[input.GetLength(0)][];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    temp[x][y] = ReLu(input[x][y], Alpha);
                }

            }
            return temp;
        }
        public static dynamic init2Ddata(dynamic data,int value)
        {
            float[][][,] gg;
            if (data is float[][][,])
            {
                for (int i = 0; i < data.Length; i++)
                {
                   
                    for (int j = 0; j < data[i].Length; j++)
                    {
                        for (int a = 0; a < data[i][j].GetLength(0); a++)
                        {

                            for (int s = 0; s < data[i][j].GetLength(1); s++)
                            {
                                data[i][j][a, s] = value;
                            }
                        }
                    }
                }
            }
            if (data is float[,])
            {
                for (int a = 0; a < data.GetLength(0); a++)
                {

                    for (int s = 0; s < data.GetLength(1); s++)
                    {
                        data[a, s] = value;
                    }
                }
            }
                return data;
        }

        public static float[,] float1DTofloat2D(float[] aa, int w, int h)
        {
            int len = aa.Length;
            float[,] data = new float[w, h];

            for (int i = 0; i < len; i++)
            {
                int col = i % h;
                int row = i / w;
                data[row, col] = aa[i];
            }
            return data;
        }

        public static dynamic zroe2D(params int[] length)
        {
            dynamic data = null;
            if (length.Length == 1)
                data = new float[length[0]];
            if (length.Length == 2)
            {
                data = new float[length[0]][];
                int len = length[0];
                for (int i = 0; i < len; i++)
                {
                    data[i] = new float[length[1]];
                }
            }
            if (length.Length == 3)
            {
                data = new float[length[0]][,];
                int len = length[0];
                for (int i = 0; i < len; i++)
                {
                    data[i] = new float[length[1], length[2]];
                    //for (int j = 0; j < length[1]; j++)
                    //{
                    //    data[i][j] = new float[length[2]];
                    //}
                }

            }
            if (length.Length == 4)
            {
                data = new float[length[0]][][,];
                int len = length[0];
                for (int i = 0; i < len; i++)
                {
                    data[i] = new float[length[1]][,];
                    for (int j = 0; j < length[1]; j++)
                    {
                        data[i][j] = new float[length[2], length[3]];
                        
                    }
                }

            }
            return data;
        }
        public static dynamic zroe(params int[] length)
        {
            dynamic data=null;
            if (length.Length == 1)
                data =new float[length[0]];
            if (length.Length == 2)
            {
                data = new float[length[0]][];
                int len = length[0];
                for (int i = 0; i < len; i++)
                {
                    data[i]=new float[ length[1]];
                }
            }
            if (length.Length == 3)
            {
                data = new float[length[0]][][];
                int len = length[0];
                for (int i = 0; i < len; i++)
                {
                    data[i] = new float[length[1]][];
                    for (int j = 0; j < length[1]; j++)
                    {
                        data[i][j] = new float[length[2]];
                    }
                }

            }
            if (length.Length == 4)
            {
                data = new float[length[0]][][][];
                int len = length[0];
                for (int i = 0; i < len; i++)
                {
                    data[i] = new float[length[1]][][];
                    for (int j = 0; j < length[1]; j++)
                    {
                        data[i][j] = new float[length[2]][];
                        for (int x = 0; x < length[2]; j++)
                        {
                            data[i][j][x] = new float[length[3]];
                        }
                    }
                }

            }
            return data;
        }

        public static float[][] copy(float[][] prev_state)
        {
            int len = prev_state.Length;
            float[][] data = new float[len][];
           
            for (int i = 0; i < len; i++)
            {
                int len2 = prev_state[i].Length;
                data[i] = new float[len2];
                for (int j = 0; j < len2; j++)
                {
                    data[i][j] = prev_state[i][j];
                }
            }
            return data;
        }

        public static float[][] cat(float[][] input, float[][] prev_hidden, int index=0)
        {
            float[][] data;
            if (index == 0)
            {
                data =new float[input.Length + prev_hidden.Length][];
                for (int i = 0; i < input.Length; i++)
                    data[i] = input[i];
                int len = input.Length;
                for (int i = len; i < (len + prev_hidden.Length); i++)
                    data[i] = prev_hidden[i- len];
                return data;
            }
            if (index == 1)
            {
                data = new float[input.Length][];
                for (int i = 0; i < input.Length; i++)
                {
                    data[i] = new float[input[i].Length + prev_hidden[i].Length];
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        data[i][j] = input[i][j];
                    }
                    int len = input[i].Length;
                    for (int j = len; j < (len + prev_hidden[i].Length); j++)
                    {
                        data[i][j] = prev_hidden[i][j-len];
                    }
                }
                return data;
            }
            return null;
        }
        public static float[][][,] cat(float[][][,] input, float[][][,] prev_hidden, int index = 0)
        {
            float[][][,] data;
            if (index == 0)
            {
                data = new float[input.Length + prev_hidden.Length][][,];
                for (int i = 0; i < input.Length; i++)
                {
                    data[i] = new float[input[i].Length][,];
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        data[i][j] = input[i][j];
                    }
                }

                int len = input.Length;
                for (int i = len; i < (len + prev_hidden.Length); i++)
                {
                    data[i] = prev_hidden[i - len];
                }
                return data;
            }
            if (index == 1)
            {
                data = new float[input.Length][][,];
                for (int i = 0; i < input.Length; i++)
                {
                    data[i] = new float[input[i].Length + prev_hidden[i].Length][,];
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        data[i][j] = input[i][j];
                    }
                    int len = input[i].Length;
                    for (int j = len; j < (len + prev_hidden[i].Length); j++)
                    {
                        data[i][j] = prev_hidden[i][j - len];
                    }
                }
                return data;
            }
            return null;
        }

        public static float Max(float[] x)
        {
            int len = x.Length;
            float max = 0;
            for (int i = 0; i < len; i++)
            {
                if (x[i] > max)
                    max = x[i];


            }
            return max;
        }
        public static float Max(float[,] x)
        {
            int len = x.GetLength(0);
            int lenj = x.GetLength(1);
            float max = 0;
            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < lenj; j++)
                    if (x[i,j] > max)
                    max = x[i,j]; 

            }
            return max;
        }
        public static float Max(float[][] x)
        {
            int len = x.GetLength(0);
            int lenj = x[0].GetLength(0);
            float max = 0;
            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < lenj; j++)
                    if (x[i][ j] > max)
                        max = x[i][ j];

            }
            return max;
        }
        public static float[,] ReLu( float[,] Ma, float Alpha)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);


            
            float[,] c = new float[m,n];
            float[,] a = Ma;


            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] = ReLu(a[i, j] , Alpha);
            return c;
        }
       static float ReLu(float x, float Alpha)
        {
            var keepElements = 0.0f;
         //   if (x >= 0)
                keepElements =(float)Convert.ToDouble(x >= 0);
            return x * keepElements + (Alpha * x * (1 - keepElements)); 

         //   return (float)(Math.Abs(x) + x) / 2.0f; 
        
        }
        public static float[][][,] ReLu(float[][][,] input, float Alpha)
        {
            // x = (np.abs(x) + x) / 2.0
            float[][][,] temp = new float[input.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    if (temp[x][y] == null)
                        temp[x][y] = new float[input[x][y].GetLength(0), input[x][y].GetLength(1)];
                    temp[x][y] = ReLu(input[x][y], Alpha);
                }
            }
            return temp;
        }
        static float ReLubackward(float x,float dot,float Alpha)
        {
            float keepElements = 0.0f;
            //if (x >= 0)
            // keepElements = x;
            keepElements = (float)Convert.ToDouble(x >= 0);

            return dot * (keepElements + (Alpha * (1 - keepElements)));

        }
        public static float[] ReLubackward(float[] Dx, float[] input, float F)
        {

            float[] temp = new float[input.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {


                temp[x] = ReLubackward(Dx[x],input[x],F);

            }
            return temp;
        }
        public static float[][] ReLubackward(float[][] Dx, float[][] input, float F)
        {

            float[][] temp = new float[input.GetLength(0)][];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    temp[x][y] = ReLubackward(Dx[x][y],input[x][y],F);
                }

            }
            return temp;
        }
        public static float[,] ReLubackward(float[,] Dx, float[,] Ma, float F)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);



            float[,] c = new float[m, n];
            float[,] a = Ma;


            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] = ReLubackward(Dx[i,j],a[i, j],F);
            return c;
        }
        public static float[][][,] ReLubackward(float[][][,] Dx, float[][][,] input, float F)
        {
            float[][][,] temp = new float[input.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    if (temp[x][y] == null)
                        temp[x][y] = new float[input[x][y].GetLength(0), input[x][y].GetLength(1)];
                    temp[x][y] = ReLubackward(Dx[x][y],input[x][y],F);
                }
            }
            return temp;
        }

        public float[,] upmaxxy = new float[0,0];
        public   List<poolxy> poolxies = new List<poolxy>();
        public Matrix()
        {
          
        }
        static   float sigmod(float x)
        {
            return (float) (1 / (1 + Math.Pow(Math.E, -x)));
        }
        protected static double dSigmoid(double x)
        {
            return (1 - x) * x;
        }
        public Matrix(int x,int y)
        {
            values = new float[x, y];
            float[,] maxxy = new float[x , y];
        }
        //public static float[,] uppooling(Matrix matrix,int scaleSize)
        //{

        //}
        public static float[,] convnFull(float[,] matrix, float[,] kernel,int stride,int p,bool puls=true)
        {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
            int km = kernel.GetLength(0);
            int kn = kernel.GetLength(1);
            float[,] extendMatrix = extend(matrix, stride, p, km);
            //float[,] extendMatrix = new float[(m* stride) + (2-p) * (km - 1), (n* stride) + (2-p) * (kn - 1)];
           
            //if (m == extendMatrix.GetLength(0) && n == extendMatrix.GetLength(1))
            //    puls = false;
            //if (puls)
            //{
            //    int s = 0,h=0;
            //    for (int i = 0; i < m; i+= stride)
            //    {
                   
            //        h = 0;
            //        for (int j = 0; j < n; j += stride)
            //        {
            //            int a = i + km - 1;
            //            int b = j + kn - 1;
            //            extendMatrix[a, b] = matrix[s, h];
            //            h++;
            //        }
            //        s++;
            //    }
            //}
            //else
            //    extendMatrix = matrix;


            return convnValid(extendMatrix, kernel, 1, 0);
        }

        public static float[][] dot(float[][] left, float[][] right)
        {



            float[][] data = new float[left.GetLength(0)][];

            int lens = left.GetLength(0);

            for (var i = 0; i < (lens); i++)
            {
                var lensx = left[i].GetLength(0);

                data[i] = new float[right[0].GetLength(0)];
                for (var j = 0; j < (lensx); j++)
                {

                    var lexj = right[j].GetLength(0);

                    for (var x = 0; x < (lexj); x++)
                        data[i][x] += left[i][j] * right[j][x];
                }
            }
            return data;
        }

        public static float[][,] float1DTofloat3D(float[] gg,int w,int h,int channl)
        {
            float[][,] data = new float[channl][,];
            
            for (int c = 0; c < channl; c++)
            {
                data[c] = new float[w,h];
                for (int x = 0; x< w; x++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        data[c][x, y] = gg[(c*(w*h))+(x*w)+y];
                    }

                }
                
            }
            return data;
        }

        public static float[] dot(float[] weights, float[] matrices)
        {
            float[] data = new float[weights.GetLength(0)];
            int lens = weights.GetLength(0);
            for (var i = 0; i < (lens); i++)
            {
              
                    data[i] += weights[i] * matrices[i];
                
            }
            return data;
        }

         
      
        public static Matrix[,] cat(Matrix[,] input, Matrix[,] prev_state,int  index)
        {
            Matrix[,] temp=null;
            if (index == 0) {
                int cell = input.GetLength(1) == 0 ? 1 : input.GetLength(1);
                temp = new Matrix[input.GetLength(0) + prev_state.GetLength(0), cell];
                int x = input.GetLength(0) + prev_state.GetLength(0);
                for (int i = 0; i < input.GetLength(0); i++)
                {
                    for (int j = 0; j < input.GetLength(1); j++)
                    {
                        temp[i, j] = input[i, j];
                    }
                }
                for (int i = x - prev_state.GetLength(0); i < x ; i++)
                {
                    for (int j = 0; j < cell; j++)
                    {
                        temp[i, j] = prev_state[i - input.GetLength(0), j];
                    }
                }
            }

            if (index == 1)
            {
                temp = new Matrix[input.GetLength(0), input.GetLength(1) + prev_state.GetLength(1)];
                int x = input.GetLength(0) ;
                int y = input.GetLength(1) + prev_state.GetLength(1);
                for (int i = 0; i < input.GetLength(0); i++)
                {
                    for (int j = 0; j < input.GetLength(1); j++)
                    {
                        temp[i, j] = input[i, j];
                    }
                }
                for (int i = 0; i < x; i++)
                {
                    for (int j = y - prev_state.GetLength(1); j < y; j++)
                    {
                        temp[i, j] = prev_state[i, j- (y - prev_state.GetLength(1))];
                    }
                }

            }
        
            return temp;

        }

        public static Matrix[] dropout(Matrix[] input)
        {
            Matrix[] temp = null;

            temp = new Matrix[input.GetLength(0) ];

            int y = input.GetLength(0) ;

            for (int j = 0; j < y ; j++)
            {
                temp[j] = new Matrix();
                temp[j].values = dropout( input[j].values,0.5f);
            } 
            return temp;

        }
        public static float[][][,] dropout(float[][][,]  input, float level)
        {
            float[][][,] temp = new float[input.Length][][,];



            int y = input.Length;
            
            for (int j = 0; j < y; j++)
            {
                int a = input[j].Length;
                temp[j] = new float[a][,];
                for (int i = 0; i < a; i++)
                {
                    temp[j][i] = new float[input[j][i].GetLength(0), input[j][i].GetLength(1)];
                    temp[j][i] = dropout(input[j][i], level);
                }
            }
            return temp;

        }
        public static float[,] dropout( float[,] x, float level)
        {
            if (level < 0 || level >= 1)
            {
                return null;
            }
            float[,] xy = new float[x.GetLength(0), x.GetLength(1)];
            ///raise Exception('Dropout level must be in interval [0, 1[.')
            var retain_prob = 1.0f - level;
            // 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
            //# 硬币 正面的概率为p，n表示每个神经元试验的次数
            //# 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
            var sample = binomial(1, retain_prob, x.Length);//#即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
                                                            //print sample
            for (int i = 0; i < x.GetLength(0); i++)
            {
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    xy[i, j] = sample;
                    x[i,j] *= sample;//#0、1与x相乘，我们就可以屏蔽某些神经元，让它们的值变为0
                                   //print x
                    x[i,j] /= retain_prob;
                }
            }
            return xy;
        }

       static float binomial(int N, float p, int k)
        {
            if (N < 0 || k < 0) return 0.0f;
            float[][] ret = new float[N + 1][];
            for (int i = 0; i < N + 1; ++i)
                ret[i] = new float[k + 1];
            //1完成递归版的2
            ret[0][0] = 1.0f;
            //2完成递归版的1
            for (int i = 1; i < N + 1; ++i)
                ret[i][0] = (1.0f - p) * ret[i - 1][0];
            for (int j = 1; j < k + 1; ++j)
                ret[0][j] = 0.0f;
            //3完成递归版的3
            for (int i = 1; i < N + 1; ++i)
                for (int j = 1; j < k + 1; ++j)
                    ret[i][j] = (1.0f - p) * ret[i - 1][j] + p * ret[i - 1][j - 1];
            return ret[N][k];
        }
        public static float[][][,] activation_tanh(float[][][,] input)
        {

            float[][][,] temp = new float[input.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    if (temp[x][y] == null)
                        temp[x][y] = new float[input[x][y].GetLength(0), input[x][y].GetLength(1)];
                    temp[x][y] = activation_tanh(input[x][y]).values;
                }
            }
            return temp;
        }
        public static Matrix[,] activation_tanh(Matrix[,] input)
        {
            for (var x = 0; x < input.GetLength(0); x++)
            {
                for (var y = 0; y < input.GetLength(1); y++)
                {
                    input[x, y] = activation_tanh(input[x, y].values);
                }
            }
            return input;
        }
        public static Matrix[] activation_tanh(Matrix[] input)
        {
            for (var x = 0; x < input.GetLength(0); x++)
            {
              
                    input[x] = activation_tanh(input[x].values);
                
            }
            return input;
        }
        public static float[] activation_tanh(float[] input)
        {

            float[] temp = new float[input.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {


                temp[x] = Tanh(input[x]);

            }
            return temp;
        }
        public static float[][] activation_tanh(float[][] input)
        {

            float[][] temp = new float[input.GetLength(0)][];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    temp[x][y] = Tanh(input[x][y]);
                }

            }
            return temp;
        }
        public static Matrix activation_tanh(float[,] Ma, float bias = 0)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);


            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            float[,] a = Ma;


            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] = Tanh(a[i, j] + bias);
            return Mc;
        }

        protected static float Tanh(float x)
        {
            return (float) Math.Tanh(x);
        }
        protected static float dTanh(float x)
        {
            return 1 - x * x;
        }
        public static Matrix[,] activation_Sigma(Matrix[,] input)
        {

            Matrix[,] temp = new Matrix[input.GetLength(0), input.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                for (var y = 0; y < input.GetLength(1); y++)
                {
                    if (temp[x, y] == null)
                        temp[x, y] = new Matrix();
                    temp[x, y] = activation_Sigma(input[x, y].values);
                }
            }
            return temp;
        }
        public static float[][][,] activation_Sigma(float[][][,] input)
        {

            float[][][,] temp = new float[input.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                 temp[x] = new float[input[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    if (temp[x][y] == null)
                        temp[x][y] = new float[input[x][y].GetLength(0), input[x][y].GetLength(1)];
                    temp[x][y] = activation_Sigma(input[x][ y]).values;
                }
            }
            return temp;
        }

        public static float[] activation_Sigma(float[] input)
        {

            float[] temp = new float[input.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
               
                    
                    temp[x] = sigmod(input[x]);
               
            }
            return temp;
        }
        public static float[][] activation_Sigma(float[][] input)
        {

            float[][] temp = new float[input.GetLength(0)][];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    temp[x][y] = sigmod(input[x][y]);
                }

            }
            return temp;
        }
        public static Matrix[] activation_Sigma(Matrix[] input)
        {

            Matrix[] temp = new Matrix[input.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
               
                    if (temp[x] == null)
                        temp[x] = new Matrix();
                    temp[x] = activation_Sigma(input[x].values);
                
            }
            return temp;
        }
        public static List<float[][]> chunk(float[][] input, int count)
        {
            int agvlen = input.Length / count;
            List<float[][]> list = new List<float[][]>();
            for (int i = 0; i < count; i++)
            {
                float[][] data = new float[agvlen][];
                for (int j = 0; j < agvlen; j++)
                {
                    int inpindex = (j) + (i* agvlen);
                    data[j] = new float[input[inpindex].Length];
                    for (int x = 0; x < input[inpindex].Length; x++)
                        data[j][x] = input[inpindex][x];
                   
                }
                list.Add(data);
            }


            return list;

        }
        public static List<float[][]> chunk(float[][] input, int count,int index)
        {
            if (index == 0)
            {
                return chunk(input, count);
            }
            else if (index == 1)
            {
                int agvlen = input[0].Length / count;
                List<float[][]> list = new List<float[][]>();

                for (int j = 0; j < count ; j++)
                {
                    float[][] data = new float[input.Length][];
                    for (int i = 0; i < input.Length; i++)
                    {
                        data[i] = new float[agvlen];
                        for (int x = 0; x < agvlen; x++)
                        {


                            data[i][x] = input[i][(j * agvlen) + x];

                        }
                    }
                    list.Add(data);
                }
                  
                
                return list;
            }


            return null;

        }
        public static List<float[][][,]> chunk(float[][][,] input, int count,int index=0)
        {

            List<float[][][,]> list = new List<float[][][,]>();
            if (index == 0)
            {
                int agvlen = input.Length / count;
              
                for (int i = 0; i < count; i++)
                {
                    float[][][,] data = new float[agvlen][][,];
                    for (int j = 0; j < agvlen; j++)
                    {
                        int inpindex = (j) + (i * agvlen);
                        data[j] = new float[input[inpindex].Length][,];
                        for (int x = 0; x < input[inpindex].Length; x++)
                            data[j][x] = input[inpindex][x];

                    }
                    list.Add(data);
                }

            }
            if (index == 1)
            {
                int agvlen = input[0].Length / count;
                for (int j = 0; j < count; j++)
                {
                    float[][][,] data = new float[input.Length][][,];
                    for (int i = 0; i < input.Length; i++)
                    {
                        data[i] = new float[agvlen][,];
                        for (int x = 0; x < agvlen; x++)
                        {


                            data[i][x] = input[i][(j * agvlen) + x];

                        }
                    }
                    list.Add(data);
                }
            }
            return list;
        }
        public static Matrix[][,] chunk(Matrix[,] input,int count,int index)
        {
            Matrix[][,] temp  =new Matrix[count][,] ;
            for (int x = 0; x < count; x++)
            {
                if (index == 0)
                {
                    temp[x] = new Matrix[input.GetLength(0) / count, input.GetLength(1)];
                    for (var i = x* count; i < (input.GetLength(0) / count) + (x * count); i++)
                    {
                        for (var j = 0; j < input.GetLength(1); j++)
                        {
                            temp[x][i - (x * count), j] = input[i, j];
                        }

                    }
                }
                if (index == 1)
                {
                    temp[x] = new Matrix[input.GetLength(0), input.GetLength(1) / count];
                    for (var i = 0; i < input.GetLength(0) ; i++)
                    {
                        for (var j = x* (input.GetLength(1) / count); j < (input.GetLength(1) / count)+ (x * (input.GetLength(1) / count)); j++)
                        {
                            temp[x][i , j - (x * (input.GetLength(1) / count))] = input[i, j];
                        }

                    }

                }
             
            }

            return temp;

        }

        public static float sum(Matrix matrix)
        {
            float temp =0;
            for (var x = 0; x < matrix.values.GetLength(0); x++)
            {
                for (var y = 0; y < matrix.values.GetLength(1); y++)
                {
                     
                    temp+= matrix.values[x,y];
                }
            }
            return temp;
        }
        public static float sum(float[][] matrix)
        {
            float temp = 0;
            for (var x = 0; x < matrix.GetLength(0); x++)
            {
                for (var y = 0; y < matrix[0].GetLength(0); y++)
                {

                    temp += matrix[x][ y];
                }
            }
            return temp;
        }
        public static float sum(float[] matrix)
        {
            float temp = 0;
            for (var x = 0; x < matrix.GetLength(0); x++)
            {
                
                    temp += matrix[x];
                
            }
            return temp;
        }
        public unsafe static float sum(float[,] matrix)
        {
            float temp = 0;
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
            fixed (float* arr = &matrix[0, 0])
            {
                for (var x = 0; x <m; x++)
                {
                    for (var y = 0; y < n; y++)
                    {

                        temp += *(arr + y + (x * m));//  matrix[x, y];
                    }
                }
            }
            return temp;
        }
        public static float[,] convnValid(float[,] matrix, float[,] kernel, int stride,int p)
        {
            
            return convolution(matrix, kernel, stride, p);
            //		kernel = rot180(kernel);
            //int m = matrix.GetLength(0);
            //int n = matrix.GetLength(1);
            //int km = kernel.GetLength(0);
            //int kn = kernel.GetLength(1);
            ////var row = ((x - x2) + 2 * p) / stride + 1;
            ////var col = ((y - y2) + 2 * p) / stride + 1;
            //int kns = ((n - kn) + 2 * p) / stride + 1;// n - kn + 1;
            //int kms = ((m - km) + 2 * p) / stride + 1;//m - km + 1;
            //float[,] outMatrix = new float[kms, kns];
          
            //for (int i = 0; i < kms; i++)
            //{
            //    for (int j = 0; j < kns; j++)
            //    {
            //        float sum = 0.0f;
            //        for (int ki = 0; ki < km; ki++)
            //        {
            //            for (int kj = 0; kj < kn; kj++)
            //                sum += matrix[i + ki, j + kj] * kernel[ki, kj];
            //        }
            //        outMatrix[i, j] = sum;

            //    }
            //}

            var x = matrix.GetLength(0);
            var y = matrix.GetLength(1);
            var x2 = kernel.GetLength(0);
            var y2 = kernel.GetLength(1);
           
            //Ho=(H−F+2×P)/S+1
            var row = ((x - x2) + 2 * p) / stride + 1;
            var col = ((y - y2) + 2 * p) / stride + 1;
            float[,] temp = new float[row, col];
            var nx = 0;

            for (var i = 0 - p; i <= x - x2 + p; i = i + stride)
            {
                var ny = 0;
                if(i>=0)
                for (var j = 0 - p; j <= y - y2 + p; j = j + stride)
                {
                    if (j>=0)
                        //var sum = 0.0f;
                        for (var i2 = 0; i2 < x2; i2++)
                            for (var j2 = 0; j2 < y2; j2++)
                            {
                                if (i + i2 < 0 || j + j2 < 0 || i + i2 >= x || j + j2 >= y)
                                { }
                                else
                                    temp[nx, ny] += matrix[i + i2, j + j2] * kernel[i2, j2];

                            }
                    //temp[nx, ny] = Math.Max(ReLU, temp[nx, ny] + bias);
                    ny++;
                }
                nx++;
            }

            return temp;
          //  return outMatrix;
        }

      public  static float[,] normalizedLog(float[,] values)
        {
            int m = values.GetLength(0);
            int n = values.GetLength(1);
            float[,] outMatrix = new float[m, n ];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {//atan(x) * 2 / π
                   // outMatrix[i, j] = (float)(Math.Atan(values[i, j])*2/Math.PI);
                    outMatrix[i,j] =(float)Math.Log10( values[i, j]);
                }
            }
            return outMatrix;
        }
        public static float[,] normalizedMinMax(float[,] values)
        {
            int m = values.GetLength(0);
            int n = values.GetLength(1);
            float[,] outMatrix = new float[m, n];
            float Min = float.MaxValue; float Max = float.MinValue;
            //x' = (x - X_min) / (X_max - X_min)
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {//atan(x) * 2 / π
                    if (values[i, j] > Max)
                        Max = values[i, j];
                    if (values[i, j] <Min )
                        Min = values[i, j];

                    // outMatrix[i,j] =(float)Math.Log10( values[i, j]);
                }
            }
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {//atan(x) * 2 / π
                    outMatrix[i, j] = (values[i, j] - Min) / (Max - Min);
                    // outMatrix[i,j] =(float)Math.Log10( values[i, j]);
                }
            }
            return outMatrix;
        }
      

        public static float[,] normalizedAtan(float[,] values)
        {
            int m = values.GetLength(0);
            int n = values.GetLength(1);
            float[,] outMatrix = new float[m, n];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {//atan(x) * 2 / π
                    outMatrix[i, j] = (float)(Math.Atan(values[i, j]) * 2 / Math.PI);
                    // outMatrix[i,j] =(float)Math.Log10( values[i, j]);
                }
            }
            return outMatrix;
        }
        public static Matrix MaxPooling(float[,] map, int stride)
        {
            Matrix ma = new Matrix();
            ma.values = map;
            return MaxPooling(ma, stride);
        }
        public static float[,] averPooling(float[,] matrix, int scale)
        {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
             int sm = m / scale;
             int sn = n / scale;
             float[,] outMatrix = new float[sm,sn];
            if (sm * scale != m || sn * scale != n)
                throw new Exception("scale matrix");
             int size = scale * scale;
            for (int i = 0; i < sm; i++)
            {
                for (int j = 0; j < sn; j++)
                {
                    float sum = 0.0f;
                    for (int si = i * scale; si < (i * scale)+ scale; si++)
                    {
                        for (int sj = j * scale; sj < (j * scale)+scale; sj++)
                        {
                            sum += matrix[si,sj];
                        }
                    }
                    outMatrix[i,j] = sum / size;
                }
            }
            return outMatrix;
        }
       
        public static Matrix MaxPooling(Matrix map, int stride)
        {
            Matrix pool = new Matrix();
            var i = map.values.GetLength(0);
            var j = map.values.GetLength(1);

            var rx = i;
            var ry = j;
            pool  = new Matrix(rx / stride, ry / stride);
            pool .upmaxxy = new float[rx, ry];
            int xx2 = 0;
            int yy2 = 0;
            for (i = 0; i < rx; i = i + stride)
            {

                if (pool .values.GetLength(0) > xx2)
                {
                    int tx = 0, ty = 0;
                    for (j = 0; j < ry; j = j + stride)
                    {
                        if (pool .values.GetLength(1) > yy2)
                        {
                            float max = 0.0f;
                            for (var xx = 0; xx < stride; xx++)
                                for (var yy = 0; yy < stride; yy++)
                                {
                                    if (map.values[i + xx, j + yy] > max)
                                    {
                                        tx = i + xx;
                                        ty = j + yy;
                                    }
                                    max = Math.Max(max, map.values[i + xx, j + yy]);

                                }

                            pool .values[xx2, yy2] = max;
                            pool .upmaxxy[tx, ty] = max;
                            poolxy pxy = new poolxy();
                            pxy.x = tx;
                            pxy.y = ty;
                            pxy.nx = xx2;
                            pxy.ny = yy2;
                            //     pxy.value = max;
                            pool .poolxies.Add(pxy);
                            yy2++;
                        }

                    }
                    //

                }
                yy2 = 0;
                xx2++;
            }
            return pool;
        }

        public static Matrix[,] MaxPooling(Matrix[,] map, int stride)
        {
            var row = map.GetLength(0);
            var col = map.GetLength(0);
            Matrix[,] pool = new Matrix[row, col];

            for (var x = 0; x < row; x++)
            {
                for (var y = 0; y < col; y++)
                {
                    var i = map[x, y].values.GetLength(0);
                    var j = map[x, y].values.GetLength(1);
                  
                    var rx = i;
                    var ry = j;
                    pool[x, y] = new Matrix(rx / stride, ry / stride);
                    pool[x, y].upmaxxy = new float[rx, ry];
                    int xx2 = 0;
                    int yy2 = 0;
                    for (i = 0; i < rx; i = i + stride)
                    {

                        if (pool[x, y].values.GetLength(0) > xx2)
                        {
                            int tx = 0, ty = 0;
                            for (j = 0; j < ry; j = j + stride)
                            {
                                if (pool[x, y].values.GetLength(1) > yy2)
                                {
                                    float max = 0.0f;
                                    for (var xx = 0; xx < stride; xx++)
                                        for (var yy = 0; yy < stride; yy++)
                                        {
                                            if (map[x, y].values[i + xx, j + yy] > max)
                                            {
                                                tx = i + xx;
                                                ty = j + yy;
                                            }
                                            max = Math.Max(max, map[x, y].values[i + xx, j + yy]);
                                            
                                        }

                                    pool[x, y].values[xx2, yy2] = max;
                                    pool[x, y].upmaxxy[tx, ty] = max;
                                    poolxy pxy = new poolxy();
                                    pxy.x = tx;
                                    pxy.y = ty;
                                    pxy.nx = xx2;
                                    pxy.ny = yy2;
                                    //     pxy.value = max;
                                    pool[x, y].poolxies.Add(pxy);
                                   yy2++;
                                }
                               
                            }
                            //

                        }
                        yy2 = 0;
                        xx2++;
                    }
                }
            }

            return pool;
        }
        public static float[,] kroneckerMax(float[,] matrix, int scale, List<poolxy> poolxiestemp)
        {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
            float[,] outMatrix = new float[m * scale, n * scale];

            foreach (poolxy pxy in poolxiestemp)
            {
                outMatrix[pxy.x, pxy.y] = matrix[pxy.nx, pxy.ny];
            }
            return outMatrix;
        }
        public static float[,] kroneckerAvg(float[,] matrix, int scale)
        {
              int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
            float[,] outMatrix = new float[m * scale,n * scale];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int ki = i * scale; ki < (i + 1) * scale; ki++)
                    {
                        for (int kj = j * scale; kj < (j + 1) * scale; kj++)
                        {
                            outMatrix[ki,kj] = matrix[i,j]/ (scale* scale);
                        }
                    }
                }
            }
            return outMatrix;
        }
        public static float[] MatrixSub(float[] ma, float[] mb)
        {

            int m = ma.GetLength(0);
          
            float[] mc = new float[m];
            for (int i = 0; i < m; i++)
            {
              
                    float output_value = mb[i];
                    mc[i] = (ma[i] - output_value);
                
            }
            return mc;
        }
        public static float[] MatrixSub(float[] ma, float mb)
        {

            int m = ma.GetLength(0);

            float[] mc = new float[m];
            for (int i = 0; i < m; i++)
            {

               
                mc[i] = (ma[i] - mb);

            }
            return mc;
        }
        public static float[][] MatrixSub(float[][] ma, float[][] mb)
        {

            int m = ma.GetLength(0);
            int n = ma[0].GetLength(0);
            float[][] mc = new float[m][];
            for (int i = 0; i < m; i++)
            {
                mc[i] = new float[n];
                for (int j = 0; j < n; j++)
                {
                    float output_value = mb[i][j];
                    mc[i][j] = (ma[i][j] - output_value);
                }
            }
            return mc;
        }
        public static float[][] MatrixSub(float ma, float[][] mb)
        {

            int m = mb.GetLength(0);
            int n = mb[0].GetLength(0);
            float[][] mc = new float[m][];
            for (int i = 0; i < m; i++)
            {
                mc[i] = new float[n];
                for (int j = 0; j < n; j++)
                {
                    float output_value = mb[i][j];
                    mc[i][j] = (ma - output_value);
                }
            }
            return mc;
        }
        public static float[][][,] MatrixSub(float[][][,] ma, float[][][,] mb)
        {

            int m = ma.GetLength(0);
            int n = ma[0].GetLength(0);
            float[][][,] mc = new float[m][][,];
            for (int i = 0; i < m; i++)
            {
                mc[i] = new float[n][,];
                for (int j = 0; j < n; j++)
                {
                    mc[i][j] = MatrixSub(ma[i][j], mb[i][j]); 
                }
            }
            return mc;
        }

        public static Matrix MatrixSub(Matrix ma, Matrix mb)
        {
             
            int m = ma.values.GetLength(0);
            int n = ma.values.GetLength(1);
            Matrix mc = new Matrix(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float output_value = mb.values[i, j];
                    mc.values[i, j] = (ma.values[i, j] - output_value);
                }
            return mc;
        }
        public static float[,] MatrixSub(float[,] ma, float[,] mb)
        {

            int m = ma.GetLength(0);
            int n = ma.GetLength(1);
            float[,] mc = new float[m, n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float output_value = mb[i, j];
                    mc[i, j] = (ma[i, j] - output_value);
                }
            return mc;
        }
        static float getRELUGradFromY(float x,float RELU)
        {
            if (x > RELU) return 1.0f;
            else return RELU;
        }
        static float getmax(float x, float RELU)
        {
            if (x > RELU) return RELU;
            else return x;
        }
        public  static float[,] MatrixMAX(float[,] matrix, float max)
        {
            var x = matrix.GetLength(0);
            var y = matrix.GetLength(1);
            float[,] m = new float[x, y];
            for (var i = 0; i < x; i++)
                for (var j = 0; j < y; j++)
                    m[i, j] = getmax(matrix[i, j], max);
            return m;
        }
        public static Matrix[] Clone(Matrix[] matrix)
        {
            var x = matrix.GetLength(0);
          
            Matrix[] m = new Matrix[x];
            for (var i = 0; i < x; i++)
                    m[i] = matrix[i].Clone();
            return m;
        }
        public static Matrix[,] Clone(Matrix[,] matrix)
        {
            var x = matrix.GetLength(0);
            var y = matrix.GetLength(1);
            Matrix[,] m = new Matrix[x, y];
            for (var i = 0; i < x; i++)
                for (var j = 0; j < y; j++)
                    m[i, j] = matrix[i, j].Clone();
            return m;
        }
        public static float[,] Clone(float[,] matrix)
        {
            var x = matrix.GetLength(0);
            var y = matrix.GetLength(1);
            float[,] m = new float[x,y];
            for (var i = 0; i < x; i++)
                for (var j = 0; j < y; j++)
                    m[i, j] = matrix[i, j];
            return m;
        }
        public static float[][] T(float[][] matrix)
        {
            float[][] ma = new float[matrix[0].Length][];
            int m = matrix[0].GetLength(0) ;
            int n = matrix.GetLength(0);
            for (int i = m-1; i >=0; i--)
            {
                ma[i] = new float[n];
                for (int j = 0; j < n ; j++)
                {
                    ma[i][j] = matrix[j][i];
                   
                  
                }
            }
            
            return ma;
        }
        public static float[][] rot180(float[][] matrix)
        {
            float[][] ma = matrix;
            int m = ma.GetLength(0);
            int n = matrix[0].GetLength(0);

            for (int i = 0; i < m; i++)
            {
               
                for (int j = 0; j < n / 2; j++)
                {
                   
                    float tmp = ma[i][ j];
                    ma[i][ j] = ma[i][ n - 1 - j];
                    ma[i][ n - 1 - j] = tmp;
                }
            }
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m / 2; i++)
                {
                    float tmp = ma[i][ j];
                    ma[i][j] = ma[m - 1 - i][ j];
                    ma[m - 1 - i][ j] = tmp;
                }
            }
            return ma;
        }
        public static float[,] rot180(float[,] matrix)
        {
            float[,] ma = Clone(matrix);
            int m = ma.GetLength(0);
            int n = ma.GetLength(1);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n / 2; j++)
                {
                    float tmp = ma[i,j];
                    ma[i,j] = ma[i,n - 1 - j];
                    ma[i,n - 1 - j] = tmp;
                }
            }
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m / 2; i++)
                {
                    float tmp = ma[i,j];
                    ma[i,j] = ma[m - 1 - i,j];
                    ma[m - 1 - i,j] = tmp;
                }
            }
            return ma;
        }
        public static Matrix[] MatrixAdd(Matrix[] Ma, Matrix[] Mb)
        {
            Matrix[] input = new Matrix[Ma.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {

                input[x] = MatrixAdd(Ma[x].values, Mb[x].values);

            }
            return input;
        }
        public static Matrix[] MatrixAdd(Matrix[] Ma, float Mb)
        {
            Matrix[] input = new Matrix[Ma.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
               
                    input[x] = MatrixAdd(Ma[x].values, Mb);
                
            }
            return input;
        }
        public static Matrix[,] MatrixAdd(Matrix[,] Ma, float Mb)
        {
            Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                for (var y = 0; y < input.GetLength(1); y++)
                {

                    input[x, y] = MatrixAdd(Ma[x, y].values, Mb);
                }
            }
            return input;
        }
        public static float[][][,] MatrixAdd(float[][][,] Ma, float[][][,] Mb)
        {
            float[][][,] input = new float[Ma.Length][][,];
          //  Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                input[x] = new float[Ma[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {

                    input[x][y] = MatrixAdd(Ma[x][y], Mb[x][y]).values;
                }
            }
            return input;
        }
        public static float[][,] MatrixAdd(float[][,] Ma, float[][,] Mb)
        {
            float[][,] input = new float[Ma.Length][,];
            //  Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                input[x] = new float[Ma[x].GetLength(0), Ma[x].GetLength(1)];
               

                    input[x] = MatrixAdd(Ma[x], Mb[x]).values;
                
            }
            return input;
        }
        public static float[][] MatrixAdd(float[][] Ma, float[][] Mb)
        {
            float[][] input = new float[Ma.Length][];
            //  Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                input[x] = new float[Ma[x].GetLength(0)];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {

                    input[x][y] = Ma[x][y]+ Mb[x][y];
                }
            }
            return input;
        }
        public static float[] MatrixAdd(float[] Ma, float[] Mb)
        {
            float[] input = new float[Ma.Length];
            //  Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
               
               
                    input[x] = Ma[x] + Mb[x];
                
            }
            return input;
        }
        public static Matrix MatrixMax(float[,] Ma, float[,] Mb)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);
            int m2 = Mb.GetLength(0);
            int n2 = Mb.GetLength(1);

            if ((m != m2) || (n != n2))
            {
                Exception myException = new Exception("数组维数不匹配");
                throw myException;
            }

            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            float[,] a = Ma;
            float[,] b = Mb;

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] =Math.Max( a[i, j] ,b[i, j]) ;
            return Mc;
        }
        public unsafe static Matrix MatrixAdd(float[,] Ma, float[,] Mb,float bias=0)
        {
           
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);
            int m2 = Mb.GetLength(0);
            int n2 = Mb.GetLength(1);

            if ((m != m2) || (n != n2))
            {
                Exception myException = new Exception("数组维数不匹配");
                throw myException;
            }

            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            //float[,] a = Ma;
            //float[,] b = Mb;
            fixed (float* marr = &Ma[0, 0])
            {
                fixed (float* mbrr = &Mb[0, 0])
                {
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                            c[i, j] = *(marr+j+(i*m)) + (*(mbrr + j + (i * m))) + bias;
                }
            }
            return Mc;
        }
        public unsafe static Matrix MatrixAdd(float[,] Ma, float bias = 0)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);
         

            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            

            fixed (float* marr = &Ma[0, 0])
            {
                
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                            c[i, j] = *(marr + j + (i * m))  + bias;
                
            }
            //for (int i = 0; i < m; i++)
            //    for (int j = 0; j < n; j++)
            //        c[i, j] = a[i, j] + bias;
            return Mc;
        }
        public unsafe static float[,] MAdd(float[,] Ma, float bias = 0)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);

             
            float[,] c = new float[m,n];


            fixed (float* marr = &Ma[0, 0])
            {

                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        c[i, j] = *(marr + j + (i * m)) + bias;

            }
            //for (int i = 0; i < m; i++)
            //    for (int j = 0; j < n; j++)
            //        c[i, j] = a[i, j] + bias;
            return c;
        }
        public static Matrix activation_ReLU(float[,] Ma, float bias = 0,float relu=0)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);


            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            float[,] a = Ma;


            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] = Math.Max(a[i, j] + bias, relu);
            return Mc;
        }
        public static Matrix activation_Sigma(float[,] Ma, float bias = 0)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);
          

            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            float[,] a = Ma;
        

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] = sigmod(a[i,j] + bias);
            return Mc;
        }
        public static Matrix[,] activation_Sigmabackward(Matrix[,] input, Matrix[,] dout)
        {
            Matrix[,] temp = new Matrix[input.GetLength(0), input.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                for (var y = 0; y < input.GetLength(1); y++)
                {
                    if (temp[x, y] == null)
                        temp[x, y] = new Matrix();
                        temp[x, y] = activation_Sigmabackward(input[x, y].values, dout[x, y].values);
                }
            }
            return temp;
        }
        public static float[][][,] activation_Sigmabackward(float[][][,] input, float[][][,] dout)
        {
            float[][][,] temp = new float[input.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    if (temp[x][y] == null)
                        temp[x][y] = new float[input[x][y].GetLength(0), input[x][y].GetLength(1)];
                    temp[x][y] = activation_Sigmabackward(input[x][y], dout[x][y]).values;
                }
            }
            return temp;
        }
        public static Matrix[] activation_Sigmabackward(Matrix[] input, Matrix[] dout)
        {
            Matrix[] temp = new Matrix[input.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                
                    if (temp[x] == null)
                        temp[x] = new Matrix();
                    temp[x] = activation_Sigmabackward(input[x].values, dout[x].values);
                
            }
            return temp;
        }
        public static Matrix activation_Sigmabackward(float[,] Ma,float [,] dout)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);


            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            float[,] a = Ma;


            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] = dout[i,j]*(1-a[i, j])*a[i, j];// dout * (1 - outdata) * outdata;
            return Mc;
        }
        public static float[][]  activation_Sigmabackward(float[][] Ma, float[][] dout)
        {
            int m = Ma.GetLength(0);
            int n = Ma[0].GetLength(0);


            float[][] Mc = new float[m][];




            for (int i = 0; i < m; i++)
            {
                Mc[i] = new float[n];
                for (int j = 0; j < n; j++)
                    Mc[i][j] = dout[i][j] * (1 - Ma[i][j]) * Ma[i][j];// dout * (1 - outdata) * outdata;
            }
            return Mc;
        }
        public static float[] activation_Sigmabackward(float[] Ma, float[] dout)
        {
            int m = Ma.GetLength(0);
           


            float[] Mc = new float[m];




            for (int i = 0; i < m; i++)
            {
             
              
                    Mc[i] = dout[i] * (1 - Ma[i]) * Ma[i];// dout * (1 - outdata) * outdata;
            }
            return Mc;
        }
        public static float[][][,] activation_tanhbackward(float[][][,] input, float[][][,] dout)
        {
            float[][][,] temp = new float[input.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                temp[x] = new float[input[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    if (temp[x][y] == null)
                        temp[x][y] = new float[input[x][y].GetLength(0), input[x][y].GetLength(1)];
                    temp[x][y] = activation_tanhbackward(input[x][y], dout[x][y]).values;
                }
            }
            return temp;
        }
        public static float[] activation_tanhbackward(float[] Ma, float[] dout)
        {
            int m = Ma.GetLength(0);
             
            int m2 = dout.GetLength(0);
          


            float[] Mc = new float[m2];
            //float[][] c = Mc.values;


            for (int i2 = 0; i2 < m2; i2++)
            {    

                Mc[i2] = (float)(dout[i2] * (1.0 - Math.Pow(Ma[i2], 2))); //dout[i, j] * (1 - a[i, j]) * a[i, j];// dout * (1 - outdata) * outdata;

                  
                
            }
            return Mc;
        }
        public static float[][] activation_tanhbackward(float[][] Ma, float[][] dout)
        {
            int m2 = Ma.GetLength(0);
            int n2 = Ma[0].GetLength(0);
          

            float[][] Mc = new float[m2][];
            //float[][] c = Mc.values;


            for (int i2 = 0; i2 < m2; i2++)
            {
                Mc[i2] = new float[n2];
                for (int j2 = 0; j2 < n2; j2++)
                {


                    Mc[i2][j2] = (float)(dout[i2][j2] * (1.0 - Math.Pow(Ma[i2][j2], 2))); //dout[i, j] * (1 - a[i, j]) * a[i, j];// dout * (1 - outdata) * outdata;

                  
                }
            }
            return Mc;
        }
        public static Matrix[,] activation_tanhbackward(Matrix[,] input, Matrix[,] dout)
        {
            Matrix[,] temp = new Matrix[input.GetLength(0), input.GetLength(1)];
             
                    for (var x = 0; x < input.GetLength(0); x++)
                    {
                        for (var y = 0; y < input.GetLength(1); y++)
                        {
                            if (temp[x, y] == null)
                                temp[x, y] = new Matrix();
                            temp[x, y] = activation_tanhbackward(input[x, y].values, dout[x, y].values);
                        }
                    }
            
            return temp;
        }
        public static Matrix[] activation_tanhbackward(Matrix[] input, Matrix[] dout)
        {
            Matrix[] temp = new Matrix[input.GetLength(0)];

            for (var x = 0; x < input.GetLength(0); x++)
            {
               
                    if (temp[x] == null)
                        temp[x] = new Matrix();
                    temp[x] = activation_tanhbackward(input[x].values, dout[x].values);
                
            }

            return temp;
        }
        public static Matrix activation_tanhbackward(float[,] Ma, float[,] dout)
        {
            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);
            int m2 = dout.GetLength(0);
            int n2 = dout.GetLength(1);


            Matrix Mc = new Matrix(m2, n2);
            float[,] c = Mc.values;
          

            for (int i2 = 0; i2 < m2; i2++)
                for (int j2 = 0; j2 < n2; j2++)
                {
                    
                    c[i2, j2]=(float)(dout[i2, j2] * (1.0 - Math.Pow(Ma[i2, j2], 2))); //dout[i, j] * (1 - a[i, j]) * a[i, j];// dout * (1 - outdata) * outdata;

                  
                }
            return Mc;
        }
      

        public static Matrix[,] divide(Matrix[,] a, float b)
        {
            int hang = a.GetLength(0);
            int lie = a.GetLength(1);

            for (int i = 0; i < hang; i++)
            {
                for (int j = 0; j < lie; j++)
                {
                    a[i, j].values=divide(a[i, j].values, b);
                }
            }
            return a;
        }
        public static float[][][,] divide(float[][][,] a, float b)
        {
            int hang = a.GetLength(0);
           
            float[][][,] result = new float[hang][][,];
            for (int i = 0; i < hang; i++)
            {
               
                int lie = a[i].GetLength(0);
                result[i] = new float[lie][,];
                for (int j = 0; j < lie; j++)
                {
                    result[i][j] = divide(a[i][j], b) ;
                }
            }
            return result;
        }
        public unsafe static float[,] divide(float[,] a, float b)
        {
            int hang = a.GetLength(0);
            int lie = a.GetLength(1);
            float[,] result = new float[hang, lie];
            fixed (float* arr = &a[0, 0])
            {
                for (int i = 0; i < hang; i++)
                {
                    for (int j = 0; j < lie; j++)
                    {
                        result[i, j] = *(arr + j + (i * hang))   / b;
                    }
                }
            }
            return result;
        }
        public unsafe static float[,] divide(float[,] a, float[,] b)
        {
            int hang = a.GetLength(0);
            int lie = a.GetLength(1);
            float[,] result = new float[hang, lie];
            fixed (float* arr = &a[0, 0])
            {
                fixed (float* arrb = &b[0, 0])
                {
                    for (int i = 0; i < hang; i++)
                    {
                        for (int j = 0; j < lie; j++)
                        {
                            result[i, j] = *(arr + j + (i * hang)) / *(arrb + j + (i * hang));
                        }
                    }
                }
            }
            return result;
        }
        public static float[][] divide(float[][] a, float b)
        {
            int hang = a.GetLength(0);
            int lie = a[0].GetLength(0);
            float[][] result = new float[hang][ ];
            for (int i = 0; i < hang; i++)
            {
                result[i] = new float[lie];
                for (int j = 0; j < lie; j++)
                {
                    result[i][ j] = a[i][j] / b;
                }
            }
            return result;
        }
        public static float[] divide(float[] a, float b)
        {
            int hang = a.GetLength(0);
         
            float[] result = new float[hang];
            for (int i = 0; i < hang; i++)
            {
              
               
                    result[i] = a[i] / b;
                
            }
            return result;
        }
        public Matrix cat(float[,] VB,bool isrow=true)
        {
            int x=0, y=0;
            if (isrow)
            {
                 x = values.GetLength(0);
                 y = values.GetLength(1) + VB.GetLength(1);
            }
            Matrix m = new Matrix(x, y);
            if (isrow)
            {
                for (var i = 0; i < x; i++)
                    for (var j = 0; j < values.GetLength(1); j++)
                        m.values[i, j] = values[i, j];
                for (var i = 0; i < x; i++)
                    for (var j = values.GetLength(1); j < y; j++)
                        m.values[i, j] = VB[i, j- values.GetLength(1)];
            }
            return m;
        }
        public Matrix Clone()
        {
            var x = values.GetLength(0);
            var y = values.GetLength(1);
            Matrix m = new Matrix(x,y);
            for (var i = 0; i < x; i++)
                for (var j = 0; j < y; j++)
                    m.values[i, j] = values[i, j];
            return m;
        }

        public void randinit()
        {
            var x = values.GetLength(0);
            var y = values.GetLength(1);
            Random rand = new Random();
            for (var i = 0; i < x; i++)
                for (var j = 0; j < y; j++)
                {
                    //System.Threading.Thread.Sleep(1);
                    values[i, j] = ((float)rand.Next() / Int32.MaxValue) * 0.1f;
                }
        }
        public static float[,] randinit(float[,] bvalue,int num)
        {
            var x = bvalue.GetLength(0);
            var y = bvalue.GetLength(1);
            Random rand = new Random();
            for (var i = 0; i < x; i++)
                for (var j = 0; j < y; j++)
                {
                   System.Threading.Thread.Yield();
                    float randnum = (((float)rand.Next() / (float)Int32.MaxValue) - 0.5f) * 2; // 产生一个-1到1的随机数
                    bvalue[i, j] = randnum * (float)Math.Sqrt(6.0f / (float)(num));
                    //bvalue[i, j] = ((float)rand.Next() / Int32.MaxValue) * 0.1f;
                }
            return bvalue;
        }
        //public Matrix convolution(Matrix m,int stride,float ReLU)
        //{
        //    Matrix ma = new Matrix();
           
        //    ma.values = convolution(m.values, stride, ReLU,0);
        //    return ma;
        //}

        //public Matrix convolution(Matrix m, int stride, float ReLU,float bias)
        //{
        //    Matrix ma = new Matrix();

        //    ma.values = convolution(m.values, stride, ReLU, bias);
        //    return ma;
        //}
        //public Matrix convolution(Matrix m, int stride, float ReLU, float bias,int p)
        //{
        //    Matrix ma = new Matrix();

        //    ma.values = convolution(m.values, stride, ReLU, bias,p);
        //    return ma;
        //}
        public static Matrix[,] MatrixAdd(Matrix[,] Ma, Matrix[,] Mb)
        {
            Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                for (var y = 0; y < input.GetLength(1); y++)
                {
                    
                    input[x, y] = MatrixAdd(Ma[x, y].values, Mb[x, y].values);
                }
            }
            return input;
        }
        public static Matrix[] multiply(Matrix[] Ma, Matrix[] Mb)
        {
            Matrix[] input = new Matrix[Ma.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                
                    if (input[x] == null)
                        input[x] = new Matrix();
                    input[x].values = multiply(Ma[x].values, Mb[x].values);
                
            }
            return input;
        }

        public static Matrix[] multiply(Matrix[] Ma, float Mb)
        {
            Matrix[] input = new Matrix[Ma.GetLength(0)];
            for (var x = 0; x < input.GetLength(0); x++)
            {

                if (input[x] == null)
                    input[x] = new Matrix();
                input[x].values = multiply(Ma[x].values, Mb);

            }
            return input;
        }
        public static float[][][,] multiply(float[][][,] Ma, float[][][,] Mb)
        {
            float[][][,] input = new float[Ma.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                input[x] = new float[Ma[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {
                    
                    input[x][ y] = multiply(Ma[x][y], Mb[x][ y]);
                }
            }
            return input;
        }
        public static float[][][,] multiply(float[][][,] Ma, float Mb)
        {
            float[][][,] input = new float[Ma.GetLength(0)][][,];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                input[x] = new float[Ma[x].GetLength(0)][,];
                for (var y = 0; y < input[x].GetLength(0); y++)
                {

                    input[x][y] = multiply(Ma[x][y], Mb);
                }
            }
            return input;
        }
        public static Matrix[,] multiply(Matrix[,] Ma, Matrix[,] Mb)
        {
            Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                for (var y = 0; y < input.GetLength(1); y++)
                {
                    if (input[x, y] == null)
                        input[x, y] = new Matrix();
                    input[x, y].values = multiply(Ma[x, y].values, Mb[x,y].values);
                }
            }
            return input;
        }
        public static Matrix[,] multiply(Matrix[,] Ma, float Mb)
        {
            Matrix[,] input = new Matrix[Ma.GetLength(0), Ma.GetLength(1)];
            for (var x = 0; x < input.GetLength(0); x++)
            {
                for (var y = 0; y < input.GetLength(1); y++)
                {
                    if (input[x, y] == null)
                        input[x, y] = new Matrix();
                    input[x, y].values = multiply(Ma[x, y].values, Mb);
                }
            }
            return input;
        }
        public static Matrix multiply(Matrix Ma, float Mb)
        {
            Matrix m = new Matrix();
            m.values= multiply(Ma.values, Mb);
            return m;
        }
        public static float[][] multiply(float[][] Ma, float Mb)

        {

            int m = Ma.GetLength(0);
            int n = Ma[0].GetLength(0);



            float[][] Mc = new float[m][];



            for (int i = 0; i < m; i++)
            {
                Mc[i] = new float[n];
                for (int j = 0; j < n; j++)
                {
                    float a = Ma[i][j];

                    float b = Mb;
                    Mc[i][j] = (float)a * b;
                }
            }
            return Mc;

        }
        public static float[][] multiply(float[][] Ma, float[][] Mb)

        {

            int m = Ma.GetLength(0);
            int n = Ma[0].GetLength(0);
            int m2 = Mb.GetLength(0);
            int n2 = Mb[0].GetLength(0);

            if ((m != m2) || (n != n2))
            {
                Exception myException = new Exception("数组维数不匹配");
                throw myException;
            }

            float[][] Mc = new float[m][];



            for (int i = 0; i < m; i++)
            {
                Mc[i] = new float[n];
                for (int j = 0; j < n; j++)
                {
                    float a = Ma[i][j];

                    float b = Mb[i][j];
                    Mc[i][j] = (float)a * b;
                }
            }
            return Mc;

        }


        public static float[] multiply(float[] Ma, float Mb)

        {

            int m = Ma.GetLength(0);
          



            float[] Mc = new float[m];



            for (int i = 0; i < m; i++)
            {
               
                    float a = Ma[i];

                    float b = Mb;
                    Mc[i] = (float)a * b;
                
            }
            return Mc;

        }
        public static float[] multiply(float[] Ma, float[] Mb)

        {

            int m = Ma.GetLength(0);
          
            int m2 = Mb.GetLength(0);
           

            float[] Mc = new float[m];



            for (int i = 0; i < m; i++)
            {
                
                    float a = Ma[i];

                    float b = Mb[i];
                    Mc[i] = (float)a * b;
                
            }
            return Mc;

        }
        public static float[,] multiply(float[,] Ma, float Mb)

        {

            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);
            

          
            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;


            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float a = Ma[i, j];

                    float b = Mb;
                    c[i, j] = (float)a * b;
                }
            return Mc.values;

        }
        /// <summary>

        /// 矩阵乘法

        /// <param name="matrix1">矩阵1</param>

        /// <param name="matrix2">矩阵2</param>

        /// <returns>积</returns>

        public static  float[,] multiply(float[,] Ma, float[,] Mb)

        {

            int m = Ma.GetLength(0);
            int n = Ma.GetLength(1);
            int m2 = Mb.GetLength(0);
            int n2 = Mb.GetLength(1);

            if ((m != m2) || (n != n2))
            {
                Exception myException = new Exception("数组维数不匹配");
                throw myException;
            }

            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;


            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float a = Ma[i, j];

                    float b =   Mb[i, j];
                    c[i, j] =(float)a * b;
                }
            return Mc.values;

        }
 
        public unsafe static float[,] Conv(float[,] value, float[,] m, int stride, int padding = 0)
        {
            var x = value.GetLength(0);
            var y = value.GetLength(1);
            var x2 = m.GetLength(0);
            var y2 = m.GetLength(1);
            var p = padding;
            //Ho=(H−F+2×P)/S+1
            var row = ((x - x2) + 2 * p) / stride + 1;
            var col = ((y - y2) + 2 * p) / stride + 1;
            float[,] temp = new float[row, col];
            int rows = 0,clos=0;
          //  fixed (float* temparr = &temp[0, 0])
            {
                fixed (float* marr = &m[0, 0])
                {
                    fixed (float* arr = &value[0, 0])
                    {
                        for (var i = 0 - p; i < x; i = i + stride)
                        {
                            //  i = i + stride;
                            //  var ny = 0;
                            // int i = cc * stride;

                          
                            if (rows < row)
                            {
                                clos = 0;
                                for (var j = 0 - p; j < y; j = j + stride)
                                {
                                    if (clos < col)
                                    {
                                        for (var i2 = 0; i2 < x2; i2++)
                                            for (var j2 = 0; j2 < y2; j2++)
                                            {
                                                if (i + i2 < 0 || j + j2 < 0 || i + i2 >= x  || j + j2 >= y)
                                                { continue; }
                                                else
                                                {
                                                    // float bb = (*(arr + (j + (i * x)) + i2 + j2)) * (*(marr + j2 + (i2 * x2)));
                                                    temp[rows, clos] += (*(arr + (j + ((i + i2) * x)) + j2)) * (*(marr + j2 + (i2 * x2)));
                                                    //   * (temparr + (j + (i * x))) += (*(arr + (j + ((i + i2) * x)) + j2)) * (*(marr + j2 + (i2 * x2)));
                                                    // temp[i, j] += value[i + i2, j + j2] * m[i2, j2];
                                                }
                                            }

                                        clos++;
                                    }
                                    // temp[i, j] = (float)(temp[i, j]);
                                }


                                rows++;
                            }

                            // temp[nx, ny] = Math.Max(ReLU, temp[nx, ny] + bias);
                            // ny++;

                            //  nx++;
                        }

                    }
                }
            }
            return temp;
        }
        public static float[,] Transposeconvolution(float[,] value, float[,] m, int stride, int padding = 0,int klen=0) {
            //if (padding == 0)
            //    padding = m.GetLength(0) / 2 + m.GetLength(0)%2==0?0:1;
           
            float[,] data= extend(value, stride,padding, klen);
            //if (data.GetLength(0) % 2 == 0 && data.GetLength(1) % 2 == 0 && padding==0)
            //    data = sub(data, -1, -1);
            return  Conv(data, m, 1, 0);
        }
        static float[,] sub(float[,] value, int x, int y)
        {
            int w = value.GetLength(0);
            int h = value.GetLength(1);
            w = w + x;
            h = h + y;
            float[,] data = new float[w, h];
            for (int i = 0; i < w ; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    data[i, j] = value[i, j];
                }
            }
            return data;
        }
        static float[,] extend2(float[,] value, int padding, int ksize)
        {
            int stride = 1;
            int w = value.GetLength(0);
            int h = value.GetLength(1);
            
             
            w = w + padding*2;
            h = h + padding * 2;
            //if (w % 2 == 0 && h % 2 == 0)
            //{

            //    w = w + 1;
            //    h = h + 1;
            //    padding += 1;
            //}


        
            //padding = ksize / 2 + ksize % 2 == 0 ? 0 : 1;
            float[,] data = new float[w, h];
            int a = 0, b = 0;
            for (int i = padding; i < w - (padding); i = i + stride)
            {
                b = 0;
                for (int j = padding; j < h - (padding); j = j + stride)
                {

                    data[i, j] = value[a, b];
                    b++;
                }
                a++;
            }
            return data;
        }
        static float[,] extend(float[,] value,int stride, int padding,int ksize)
        {
            int w = value.GetLength(0);
            int h=value.GetLength(1);
            int paddingn = 1;
             if (ksize>1)
             paddingn = ksize - padding - 1;
            w = w  + (w - 1) * (stride - 1) + (paddingn * 2) ;
            h = h   + (h - 1) * (stride - 1) + (paddingn * 2);
            //if (w % 2 == 0 && h % 2 == 0)
            //{

            //    w = w + 1;
            //    h = h + 1;
            //    padding += 1;
            //}


            padding = paddingn;
            //padding = ksize / 2 + ksize % 2 == 0 ? 0 : 1;
            float[,] data = new float[w,h];
            int a = 0, b = 0;
            for (int i = padding; i < w - (padding); i = i + stride)
            {
                b = 0;
                for (int j = padding; j < h- (padding); j=j+ stride)
                {
                   
                    data[i, j] = value[a,b];
                    b++;
                }
                a++;
            }
            return data;
        }
        
        public static float[,] convolution(float[,] value, float[,] m, int stride, int padding = 0)
        {
            if (CUDA)
            {
                DateTime stat = DateTime.Now;
                //  value = extend2(value, padding, 3);
                int k = m.GetLength(0);

                int w = value.GetLength(0);
                int h = value.GetLength(1);

                var row = ((w - k) + 2 * padding) / stride + 1;
                var col = ((h - k) + 2 * padding) / stride + 1;

                int S = (w - (k - 1)) * (h - (k - 1));
                float[] h_A = Matrix.float2DTofloat1D(value);
                float[] h_B = Matrix.float2DTofloat1D(m);

                CudaDeviceVariable<float> d_A = h_A;
                CudaDeviceVariable<float> d_B = h_B;
                CudaDeviceVariable<float> d_C = new float[row * col];

                int BLOCK_WIDTH = O_TILE_WIDTH + 1;


                CUDAConvKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(BLOCK_WIDTH, BLOCK_WIDTH);
                CUDAConvKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((w - 1) / O_TILE_WIDTH + 1, (h - 1) / O_TILE_WIDTH + 1);
                CUDAConvKernel.SetConstantVariable("O_TILE_WIDTH", O_TILE_WIDTH);
                CUDAConvKernel.Run(d_A.DevicePointer, d_B.DevicePointer, d_C.DevicePointer, k, w, h, row, col, stride, padding);
                float[] h_C = d_C;
                d_A.Dispose();
                d_B.Dispose();
                d_C.Dispose();
                DateTime end = DateTime.Now;
               // Console.WriteLine($"CUDA计算convolution时间：{(end - stat).TotalMilliseconds}");
                return Matrix.float1DTofloat2D(h_C, row, col);

            }
            else
                return Conv(value, m, stride, padding);
            //var x = value.GetLength(0);
            //var y = value.GetLength(1);
            //var x2 = m.GetLength(0);
            //var y2 = m.GetLength(1);
            //var p = padding;
            //Ho = (H−F + 2×P)/ S + 1
            //var row = ((x - x2) + 2 * p) / stride + 1;
            //var col = ((y - y2) + 2 * p) / stride + 1;
            //float[,] temp = new float[row, col];
            //var nx = 0;
            //Parallel.For(0 - p, (x - x2 + p) / stride, cc =>
            //for (var i = 0 - p; i <= x - x2 + p; i = i + stride)
            //{
            //    i = i + stride;
            //    var ny = 0;
            //    int i = cc * stride;
            //    if (i >= 0 && i < x)
            //        for (var j = 0 - p; j <= y - y2 + p; j = j + stride)
            //        {
            //            if (j >= 0 && j < y)
            //            {
            //                for (var i2 = 0; i2 < x2; i2++)
            //                    for (var j2 = 0; j2 < y2; j2++)
            //                    {
            //                        if (i + i2 < 0 || j + j2 < 0 || i + i2 >= x || j + j2 >= y)
            //                        { continue; }
            //                        else
            //                            temp[i, j] += value[i + i2, j + j2] * m[i2, j2];

            //                    }
            //                temp[i, j] = (float)(temp[i, j]);
            //            }
            //        }
            //    temp[nx, ny] = Math.Max(ReLU, temp[nx, ny] + bias);
            //    ny++;

            //    nx++;
            //}
            //);

           // return temp;
        }
         
        public static Matrix[,] Clip(Matrix[,] grads)
        {
            for (var x = 0; x < grads.GetLength(0); x++)
            {
                for (var y = 0; y < grads.GetLength(1); y++)
                {

                    grads[x, y] = Clip(grads[x, y]);
                }
            }
            return grads;
        }
        public static Matrix Clip(Matrix grads)
        {
            for (var x = 0; x < grads.values.GetLength(0); x++)
            {
                for (var y = 0; y < grads.values.GetLength(1); y++)
                {

                    grads.values[x, y] = Clip(grads.values[x, y]);
                }
            }
            return grads;
        }
        protected static float Clip(float x)
        {
            if (x < -1.0) return -1.0f;
            if (x > 1.0) return 1.0f;
            return x;
        }
        /// <summary>
        /// 追个后2减前1
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static float[] diff(float[] data)
        {
            float[] result = new float[data.Length-1];
            for (int i = 0; i < data.Length - 1; i++)
            {
                result[i] = data[i + 1]-data[i] ;
            }
            return result;
        }
        public static float[] pow(float[] a, float b)
        {
            int hang = a.GetLength(0);

            float[] result = new float[hang];
            for (int i = 0; i < hang; i++)
            {


                result[i] =(float) Math.Pow( a[i] , b);

            }
            return result;
        }
        public static float[] sqrt(float[] a)
        {
            int hang = a.GetLength(0);

            float[] result = new float[hang];
            for (int i = 0; i < hang; i++)
            {


                result[i] = (float)Math.Sqrt(a[i]);

            }
            return result;
        }

        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <param name="Ma"></param>
        /// <param name="Mb"></param>
        /// <returns></returns>
        public static Matrix MatrixTrans(Matrix Ma)
        {
            int m = Ma.values.GetLength(0);
            int n = Ma.values.GetLength(1);
            Matrix Mc = new Matrix(n, m);
            float[,] c = Mc.values;
            float[,] a = Ma.values;

            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    c[i, j] = a[j, i];

            return Mc;
        }
        /// <summary>
        /// 矩阵求逆(高斯法)
        /// </summary>
        /// <param name="Ma"></param>
        /// <returns></returns>
        public static Matrix MatrixInv(Matrix Ma)
        {
            int m = Ma.values.GetLength(0);
            int n = Ma.values.GetLength(1);
            if (m != n)
            {
                Exception myException = new Exception("数组维数不匹配");
                throw myException;
            }
            Matrix Mc = new Matrix(m, n);
            float[,] a0 = Ma.values;
            float[,] a = (float[,])a0.Clone();
            float[,] b = Mc.values;

            int i, j, row, k;
            float max, temp;

            //单位矩阵
            for (i = 0; i < n; i++)
            {
                b[i, i] = 1;
            }
            for (k = 0; k < n; k++)
            {
                max = 0; row = k;
                //找最大元，其所在行为row
                for (i = k; i < n; i++)
                {
                    temp = Math.Abs(a[i, k]);
                    if (max < temp)
                    {
                        max = temp;
                        row = i;
                    }

                }
                if (max == 0)
                {
                    Exception myException = new Exception("没有逆矩阵");
                    throw myException;
                }
                //交换k与row行
                if (row != k)
                {
                    for (j = 0; j < n; j++)
                    {
                        temp = a[row, j];
                        a[row, j] = a[k, j];
                        a[k, j] = temp;

                        temp = b[row, j];
                        b[row, j] = b[k, j];
                        b[k, j] = temp;
                    }

                }

                //首元化为1
                for (j = k + 1; j < n; j++) a[k, j] /= a[k, k];
                for (j = 0; j < n; j++) b[k, j] /= a[k, k];

                a[k, k] = 1;

                //k列化为0
                //对a
                for (j = k + 1; j < n; j++)
                {
                    for (i = 0; i < k; i++) a[i, j] -= a[i, k] * a[k, j];
                    for (i = k + 1; i < n; i++) a[i, j] -= a[i, k] * a[k, j];
                }
                //对b
                for (j = 0; j < n; j++)
                {
                    for (i = 0; i < k; i++) b[i, j] -= a[i, k] * b[k, j];
                    for (i = k + 1; i < n; i++) b[i, j] -= a[i, k] * b[k, j];
                }
                for (i = 0; i < n; i++) a[i, k] = 0;
                a[k, k] = 1;
            }

            return Mc;
        }
        /// <summary>
        /// 矩阵求逆(伴随矩阵法)
        /// </summary>
        /// <param name="Ma"></param>
        /// <returns></returns>
        public static Matrix MatrixInvByCom(Matrix Ma)
        {
            double d = MatrixDet(Ma);
            if (d == 0)
            {
                Exception myException = new Exception("没有逆矩阵");
                throw myException;
            }
            Matrix Ax = MatrixCom(Ma);
            Matrix An = Matrix.multiply( Ax, (float)(1.0f / d));
            return An;
        }
        /// <summary>
        /// 对应行列式的代数余子式矩阵
        /// </summary>
        /// <param name="Ma"></param>
        /// <returns></returns>
        public static Matrix MatrixSpa(Matrix Ma, int ai, int aj)
        {
            int m = Ma.values.GetLength(0);
            int n = Ma.values.GetLength(1);
            if (m != n)
            {
                Exception myException = new Exception("数组维数不匹配");
                throw myException;
            }
            int n2 = n - 1;
            Matrix Mc = new Matrix(n2, n2);
            float[,] a = Ma.values;
            float[,] b = Mc.values;

            //左上
            for (int i = 0; i < ai; i++)
                for (int j = 0; j < aj; j++)
                {
                    b[i, j] = a[i, j];
                }
            //右下
            for (int i = ai; i < n2; i++)
                for (int j = aj; j < n2; j++)
                {
                    b[i, j] = a[i + 1, j + 1];
                }
            //右上
            for (int i = 0; i < ai; i++)
                for (int j = aj; j < n2; j++)
                {
                    b[i, j] = a[i, j + 1];
                }
            //左下
            for (int i = ai; i < n2; i++)
                for (int j = 0; j < aj; j++)
                {
                    b[i, j] = a[i + 1, j];
                }
            //符号位
            if ((ai + aj) % 2 != 0)
            {
                for (int i = 0; i < n2; i++)
                    b[i, 0] = -b[i, 0];

            }
            return Mc;
        }
        /// <summary>
        /// 矩阵的行列式
        /// </summary>
        /// <param name="Ma"></param>
        /// <returns></returns>
        public static float MatrixDet(Matrix Ma)
        {
            int m = Ma.values.GetLength(0);
            int n = Ma.values.GetLength(1);
            if (m != n)
            {
                Exception myException = new Exception("数组维数不匹配");
                throw myException;
            }
            float[,] a = Ma.values;
            if (n == 1) return a[0, 0];

            float D = 0;
            for (int i = 0; i < n; i++)
            {
                D += a[1, i] * MatrixDet(MatrixSpa(Ma, 1, i));
            }
            return D;
        }

        /// <summary>
        /// 矩阵的伴随矩阵
        /// </summary>
        /// <param name="Ma"></param>
        /// <returns></returns>
        public static Matrix MatrixCom(Matrix Ma)
        {
            int m = Ma.values.GetLength(0);
            int n = Ma.values.GetLength(1);
            Matrix Mc = new Matrix(m, n);
            float[,] c = Mc.values;
            float[,] a = Ma.values;

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c[i, j] = MatrixDet(MatrixSpa(Ma, j, i));

            return Mc;
        }

 
    }
}
