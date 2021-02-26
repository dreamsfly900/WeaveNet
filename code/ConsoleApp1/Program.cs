using ManagedCuda;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
      
        static Random rand = new Random();
        static void Main(string[] args)
        {
#if LINUX
      
#else
           

#endif
            CudaContext ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId());
           int count= CudaContext.GetDeviceCount();
            
         
            string resName;
            if (IntPtr.Size == 8)
                resName = "conv2d.ptx";
            else
                resName = "conv2d.ptx";

            string resNamespace = "ConsoleApp1";
            string resource = resNamespace + "." + resName;
            string[] liste = Assembly.GetExecutingAssembly().GetManifestResourceNames();
            Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resource);
            if (stream == null) throw new ArgumentException("Kernel not found in resources.");
            int padding = 0;
            int w = 5;
            int h = 5;
            int k = 3;
            int N = k * k;
            int P = w * h;
            int S = (w - (k - 1)+ padding) * (h - (k - 1)+ padding);
            float[] h_A = new float[P];
            float[] h_B = new float[N];
            RInit(h_A, P);
            RandomInit(h_B, N);
            CudaDeviceVariable<float> d_A1 = new float[5];
 
            CudaDeviceVariable<float> d_A = h_A;
            CudaDeviceVariable<float> d_B = h_B;
            CudaDeviceVariable<float> d_C = new  float[P];
            int O_TILE_WIDTH = 16;
            int BLOCK_WIDTH = O_TILE_WIDTH + (k - 1);
            CudaKernel vectorAddKernel = ctx.LoadKernelPTX(stream, "convolution_2D_shared");
            vectorAddKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(BLOCK_WIDTH, BLOCK_WIDTH) ;
            vectorAddKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((S - 1) / O_TILE_WIDTH + 1, (S - 1) / O_TILE_WIDTH + 1)  ;
            //SetConstantVariable
             
            vectorAddKernel.SetConstantVariable("O_TILE_WIDTH", O_TILE_WIDTH );
            vectorAddKernel.Run(d_A.DevicePointer, d_B.DevicePointer, d_C.DevicePointer, k, w,h, w,h,1,1);
            float[] h_C = d_C;
            bool che = check(h_C, h_B);

            d_A.Dispose();
            d_B.Dispose();
            d_C.Dispose();
        }


      static  public bool check(float[] h_C, float[] h_B)
        {
            float sum = 0;
            for (int i = 0; i < h_B.Length; ++i)
                sum += h_B[i];
            for (int i = 0; i < h_C.Length; ++i)
                if (h_C[i] != sum)
                    return false;

            return true;

        }
        static void RandomInit(float[] data, int n)
        {
            for (int i = 0; i < n; ++i)
                data[i] = (float)rand.NextDouble();
        }
        static void RInit(float[] data, int n)
        {
            for (int i = 0; i < n; ++i)
                data[i] = (float)1;
        }
    }
}
