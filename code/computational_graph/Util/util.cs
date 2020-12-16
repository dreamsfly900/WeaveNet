using FCN;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;

namespace DenseCRF
{
    public class util
    {
      public   static void prirt(float[][][,] value)
        {
            for (var i = 0; i < value.GetLength(0); i++)
            {
                for (var j = 0; j < value[i].GetLength(0); j++)
                {
                    prirt(value[i][j]);
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }
        public static void prirt(float[,] value)
        {
            for (var i = 0; i < value.GetLength(0); i++)
            {
                for (var j = 0; j < value.GetLength(1); j++)
                {
                    Console.Write(value[i, j] + ",");
                }
                Console.WriteLine();
            }
        }
        public static float[,] readRADARMatrix(String imgpath)
        {
           
            System.IO.StreamReader sr = new System.IO.StreamReader(imgpath, Encoding.UTF8);
            string strmode = sr.ReadToEnd();
            sr.Close();
            float[,] tremp = Newtonsoft.Json.JsonConvert.DeserializeObject<float[,]>(strmode);

            for (var x = 0; x < tremp.GetLength(0); x++)
                for (var y = 0; y < tremp.GetLength(0); y++)
                {
                    if (tremp[x, y] > 0)
                    {
                         tremp[x, y] = ((tremp[x, y]) / 70);
                     //   tremp[x, y] = ((tremp[x, y]) / 70);
                    }
                  //  else { tremp[x, y] = 5 / 70; }

                }
       
            return tremp;
        }
       
        public static void prirt(float[][] value)
        {
            for (var i = 0; i < value.GetLength(0); i++)
            {
                for (var j = 0; j < value[0].GetLength(0); j++)
                {
                    Console.Write(value[i][ j] + ",");
                }
                Console.WriteLine();
            }
        }
        public static void prirt(float[] value)
        {
            for (var i = 0; i < value.GetLength(0); i++)
            {

                Console.WriteLine(value[i] + ",");


            }
        }
        public static string getstr(string file)
        {
            System.IO.StreamReader sr = new System.IO.StreamReader(file);
            string str = sr.ReadToEnd();
            sr.Close();
            return str;
        }
        public float[,] multiply(float[,] matrix1, float[,] matrix2)

        {

            //matrix1是m*n矩阵，matrix2是n*p矩阵，则result是m*p矩阵

            int m = matrix1.GetLength(0), n = matrix2.GetLength(0), p = matrix2.GetLength(1);

            float[,] result = new float[m, p];

            //矩阵乘法：c[i,j]=Sigma(k=1→n,a[i,k]*b[k,j])

            for (int i = 0; i < m; i++)

            {

                for (int j = 0; j < p; j++)

                {

                    //对乘加法则

                    for (int k = 0; k < n; k++)

                    {

                        result[i, j] += (matrix1[i, k] * matrix2[k, j]);

                    }

                }

            }

            return result;

        }
        public  static float[,,] readpng(String imgpath)
        {
            Bitmap image = new Bitmap(imgpath);
            int i, j;
            float[,,] GreyImage = new float[image.Width, image.Height,3];  //[Row,Column]
          
            BitmapData bitmapData1 = image.LockBits(new Rectangle(0, 0, image.Width, image.Height),
                                     ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                byte* imagePointer1 = (byte*)bitmapData1.Scan0;

                for (i = 0; i < bitmapData1.Height; i++)
                {
                    for (j = 0; j < bitmapData1.Width; j++)
                    {
                        GreyImage[j, i,0] = (int)imagePointer1[0];//B
                        GreyImage[j, i, 1] = (int)imagePointer1[1];//G
                        GreyImage[j, i, 2] = (int)imagePointer1[2];//R
                        //      GreyImage[j, i, 2] = (int)imagePointer1[3];//A
                        //4 bytes per pixel
                        imagePointer1 += 4;
                    }//end for j
                    //4 bytes per pixel
                    imagePointer1 += bitmapData1.Stride - (bitmapData1.Width * 4);
                }//end for i
            }//end unsafe
            image.UnlockBits(bitmapData1); 
            image.Dispose();
            return GreyImage;
        }
        //public static Matrix[,] convolutions(Matrix[] Tmap, Matrix[,] weights, int stride)
        //{
        //    var channl = Tmap.Length;
        //    var num = weights.GetLength(1);
        //    Matrix[,] map = new Matrix[channl, num];
        //    for (var k = 0; k < channl; k++)
        //    {
        //        for (var k2 = 0; k2 < num; k2++)
        //        {
        //            map[k, k2] = util.corr2d(Tmap[k], weights[k, k2], stride, 0.0f);
        //        }
        //    }
        //    return map;
        //}
        static int  converRgbToArgb(byte r, byte g, byte b)
        {
            var color = ((0xFF << 24) | (r << 16) | (g << 8) | b);
            return color;
        }
         public static Matrix[] readpnggetMatrixOne(String imgpath)
        {
            float R = 0, G = 0, B = 0;
            Matrix[] mx = new Matrix[1];
            Bitmap image = new Bitmap(imgpath);
            int i, j;
            //  float[,] GreyImage = new float[image.Width, image.Height];  //[Row,Column]
            for (i = 0; i < mx.Length; i++)
            {
                mx[i] = new Matrix(image.Width, image.Height);
            }
            BitmapData bitmapData1 = image.LockBits(new Rectangle(0, 0, image.Width, image.Height),
                                     ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                byte* imagePointer1 = (byte*)bitmapData1.Scan0;

                for (i = 0; i < bitmapData1.Height; i++)
                {
                    for (j = 0; j < bitmapData1.Width; j++)
                    {
                        if ((int)imagePointer1[3] != 0)
                        {
                            R += imagePointer1[2] / 255.0f;
                            G += imagePointer1[1] / 255.0f;
                            B += imagePointer1[0] / 255.0f;
                            mx[0].values[j, i] =(( imagePointer1[2] * 19595 + imagePointer1[1] * 38469 + imagePointer1[0] * 7472) >> 16)/255.0f;



                            //mx[0].values[j, i] = (float)imagePointer1[0] ;//B
                            //mx[1].values[j, i] = (float)imagePointer1[1] ;//G
                            //mx[2].values[j, i] = (float)imagePointer1[2] ;//R
                            //      GreyImage[j, i, 2] = (int)imagePointer1[3];//A
                            //4 bytes per pixel
                        }
                        else
                            mx[0].values[j, i] = -1;
                        imagePointer1 += 4;
                    }//end for j
                    //4 bytes per pixel
                    imagePointer1 += bitmapData1.Stride - (bitmapData1.Width * 4);
                }//end for i
            }//end unsafe
            //float agvR = R / (image.Width * image.Height);
            //float agvG = G / (image.Width * image.Height);
            //float agvB = B / (image.Width * image.Height);
            //for (i = 0; i < bitmapData1.Height; i++)
            //{
            //    for (j = 0; j < bitmapData1.Width; j++)
            //    {
            //        if (mx[0].values[j, i] != -1)
            //        {
            //            mx[0].values[j, i] = (mx[0].values[j, i] - (agvB + agvG + agvR)) / 3;//B
            //                                                                                 //   mx[0].values[j, i] = mx[0].values[j, i] < 0 ? 1 - Math.Abs(mx[0].values[j, i]) : mx[0].values[j, i];
            //        }
            //        else
            //            mx[0].values[j, i] = 0;


            //    }
            //}
            image.UnlockBits(bitmapData1);
            image.Dispose();
            return mx;
        }
        public static Matrix[] readpnggetMatrixHSB(String imgpath)
        {
            float R = 0, G = 0, B = 0;
            Matrix[] mx = new Matrix[3];
            Bitmap image = new Bitmap(imgpath);
            int i, j;
            //  float[,] GreyImage = new float[image.Width, image.Height];  //[Row,Column]
            for (i = 0; i < mx.Length; i++)
            {
                mx[i] = new Matrix(image.Width, image.Height);
            }
            BitmapData bitmapData1 = image.LockBits(new Rectangle(0, 0, image.Width, image.Height),
                                     ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                byte* imagePointer1 = (byte*)bitmapData1.Scan0;

                for (i = 0; i < bitmapData1.Height; i++)
                {
                    for (j = 0; j < bitmapData1.Width; j++)
                    {
                        if ((int)imagePointer1[3] != 0)
                        {
                         
                            Color c = System.Drawing.Color.FromArgb(imagePointer1[2], imagePointer1[1], imagePointer1[0]);
                         //   mx[0].values[j, i] = c.GetBrightness()+ c.GetHue()+ c.GetHue();
                            mx[0].values[j, i] = c.GetBrightness();//B
                            mx[1].values[j, i] = c.GetHue();//G
                            mx[2].values[j, i] = c.GetHue();//R
                            R += c.GetHue();
                            G += c.GetHue();
                            B += c.GetBrightness();

                            //mx[0].values[j, i] = (float)imagePointer1[0] ;//B
                            //mx[1].values[j, i] = (float)imagePointer1[1] ;//G
                            //mx[2].values[j, i] = (float)imagePointer1[2] ;//R
                            //      GreyImage[j, i, 2] = (int)imagePointer1[3];//A
                            //4 bytes per pixel
                        }
                        imagePointer1 += 4;
                    }//end for j
                    //4 bytes per pixel
                    imagePointer1 += bitmapData1.Stride - (bitmapData1.Width * 4);
                }//end for i
            }//end unsafe
            float agvR = R / (image.Width * image.Height);
            float agvG = G / (image.Width * image.Height);
            float agvB = B / (image.Width * image.Height);
            for (i = 0; i < bitmapData1.Height; i++)
            {
                for (j = 0; j < bitmapData1.Width; j++)
                {
                    mx[0].values[j, i] = (mx[0].values[j, i] - (agvB + agvG + agvR)) / 3;//B

                }
            }
            image.UnlockBits(bitmapData1);
            image.Dispose();
            return mx;
        }
        public static Matrix[] readpnggetMatrix(String imgpath,bool agv=true)
        {
            int R = 0, G = 0, B = 0;
            Matrix[] mx =new Matrix[3];
            Bitmap image = new Bitmap(imgpath);
            int i, j;
          //  float[,] GreyImage = new float[image.Width, image.Height];  //[Row,Column]
            for ( i = 0; i < 3; i++)
            {
                mx[i] = new Matrix(image.Width, image.Height);
            }
            BitmapData bitmapData1 = image.LockBits(new Rectangle(0, 0, image.Width, image.Height),
                                     ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                byte* imagePointer1 = (byte*)bitmapData1.Scan0;

                for (i = 0; i < bitmapData1.Height; i++)
                {
                    for (j = 0; j < bitmapData1.Width; j++)
                    {
                        if ((int)imagePointer1[3] != 0)
                        {
                            R += imagePointer1[2];
                            G += imagePointer1[1];
                            B += imagePointer1[0];
                            mx[0].values[j, i] = (float)imagePointer1[0];//B
                            mx[1].values[j, i] = (float)imagePointer1[1] ;//G
                            mx[2].values[j, i] = (float)imagePointer1[2];//R
                            //mx[0].values[j, i] = (float)imagePointer1[0] ;//B
                            //mx[1].values[j, i] = (float)imagePointer1[1] ;//G
                            //mx[2].values[j, i] = (float)imagePointer1[2] ;//R
                                                                                  //      GreyImage[j, i, 2] = (int)imagePointer1[3];//A
                                                                                  //4 bytes per pixel
                        }
                        imagePointer1 += 4;
                    }//end for j
                    //4 bytes per pixel
                    imagePointer1 += bitmapData1.Stride - (bitmapData1.Width * 4);
                }//end for i
            }//end unsafe
            if (agv)
            {
                float agvR = R / (image.Width * image.Height);
                float agvG = G / (image.Width * image.Height);
                float agvB = B / (image.Width * image.Height);
                for (i = 0; i < bitmapData1.Height; i++)
                {
                    for (j = 0; j < bitmapData1.Width; j++)
                    {
                        mx[0].values[j, i] = (mx[0].values[j, i] - agvB) / 255.0f;//B
                                                                                  //  mx[0].values[j, i] = mx[0].values[j, i] < 0 ? 1 - Math.Abs(mx[0].values[j, i]) : mx[0].values[j, i];
                        mx[1].values[j, i] = (mx[1].values[j, i] - agvG) / 255.0f;//G
                                                                                  //  mx[1].values[j, i] = mx[1].values[j, i] < 0 ? 1 - Math.Abs(mx[1].values[j, i]) : mx[1].values[j, i];
                        mx[2].values[j, i] = (mx[2].values[j, i] - agvR) / 255.0f;//R
                                                                                  //  mx[2].values[j, i] = mx[2].values[j, i] < 0 ? 1 - Math.Abs(mx[2].values[j, i]) : mx[2].values[j, i];
                    }
                }
            }
            else
            {
                for (i = 0; i < bitmapData1.Height; i++)
                {
                    for (j = 0; j < bitmapData1.Width; j++)
                    {
                        mx[0].values[j, i] = (mx[0].values[j, i] ) / 255.0f;//B
                                                                                  //  mx[0].values[j, i] = mx[0].values[j, i] < 0 ? 1 - Math.Abs(mx[0].values[j, i]) : mx[0].values[j, i];
                        mx[1].values[j, i] = (mx[1].values[j, i] ) / 255.0f;//G
                                                                                  //  mx[1].values[j, i] = mx[1].values[j, i] < 0 ? 1 - Math.Abs(mx[1].values[j, i]) : mx[1].values[j, i];
                        mx[2].values[j, i] = (mx[2].values[j, i] ) / 255.0f;//R
                                                                                  //  mx[2].values[j, i] = mx[2].values[j, i] < 0 ? 1 - Math.Abs(mx[2].values[j, i]) : mx[2].values[j, i];
                    }
                }
            }
            image.UnlockBits(bitmapData1);
            image.Dispose();


            return mx;
        }
        public static int getColor(float c, float c1, float c2)
        {
            return  (int)c + 256 * (int)c1 + 256 * 256 * (int)c2;
        }
        public static Matrix[][] initweights_(int x, int y, int z, int num)
        {
            Matrix[][] weights = new Matrix[z] [];
           
            for (var a = 0; a < z; a++)
            {
                if (weights[a] == null)
                    weights[a] = new Matrix[num];
            for (var b = 0; b < num; b++)
                {
                  
                       weights[a][ b] = new Matrix(x, y);
                    weights[a] [b].randinit();
                    System.Threading.Thread.Sleep(10);
                    //缺少随机填充初始化
                }

            }
            return weights;

        }
        public static float[][] initweights( int inc, int outnum)
        {
            float[][] weights = new float[inc][];
            Random rand = new Random();
            for (var a = 0; a < inc; a++)
            {
                weights[a] = new float[outnum];
                for (var b = 0; b < outnum; b++)
                {
                   
                    weights[a][b]= ((float)rand.Next() / Int32.MaxValue) * 0.1f;
                    System.Threading.Thread.Sleep(10);
                    //缺少随机填充初始化
                }

            }
            return weights;

        }
        public static Matrix[,] initweights(int x, int y, int z, int num)
        {
            Matrix[,] weights = new Matrix[z, num];
            for (var a = 0; a < z; a++)
            {
                for (var b = 0; b < num; b++)
                {
                    weights[a, b] = new Matrix(x, y);
                    weights[a, b].randinit();
                       System.Threading.Thread.Sleep(10);
                    //缺少随机填充初始化
                }

            }
            return weights;

        }
        public static float[][][,] initweight(int x, int y, int z, int num)
        {
            
            float[][][,] weights = new float[z][][,];
            for (var a = 0; a < z; a++)
            {
                System.Threading.Thread.Sleep(1);
                weights[a] = new float[num][,];
                for (var b = 0; b < num; b++)
                {
                    weights[a][b] = new float[x, y];
                    weights[a][b]=Matrix. randinit(weights[a][b], z+ num);
                    System.Threading.Thread.Sleep(1);
                    //缺少随机填充初始化
                }

            }
            return weights;

        }
        //public static float[,] corr2d(float[,] mx, float[,] m, int stride,float relu)
        //{
        //    Matrix max = new Matrix();
        //    max.values = mx;
        //    return max.convolution(m, stride, relu,0);
        //}
        //public static Matrix corr2d(Matrix mx,Matrix m,int stride, float relu)
        //{
        //    return mx.convolution(m, stride, relu);
        //}
        static double GT_PROB = 0.5;
        public static float[,,] classify(float[,,] anno, int M)
        {
             
            int nColors = 0;
            int[] colors=new int[255];
            int W = anno.GetLength(0); int H = anno.GetLength(1);
            float u_energy = (float)-Math.Log(1.0 / M);
            float n_energy = (float)-Math.Log((1.0f - GT_PROB) / (M - 1));
            float p_energy = (float)-Math.Log(GT_PROB);
            float[,,] res = new float[W, H, M];
            for (int k = 0; k < W; k++)
            {
                for (int j = 0; j < H; j++)
                {

                    int c = getColor(anno[k, j, 2], anno[k, j, 1], anno[k, j, 0]);
                    int i = 0;
                    for (i = 0; i < nColors && c != colors[i]; i++) ;

                    if (c != colors[i] && i == nColors)
                    {
                        if (i < M)

                            colors[nColors++] = c;
                        else
                        {
                            int min = int.MaxValue;
                            int index = -1;
                            for (int b = 0; b < nColors; b++) {
                                if (Math.Abs(colors[b] - c) < min)
                                {
                                    min = Math.Abs(colors[b] - c);
                                    index = b;
                                }
                            }
                            c = colors[index];
                            i = index;
                            //  c = 0;
                        }
                    }
                    if (c != 0)
                    {
                        for (int s = 0; s < M; s++)
                            res[k, j, s] = n_energy;
                        res[k, j, i] = p_energy;
                    }
                    else
                    {
                        for (int s = 0; s < M; s++)
                            res[k, j, s] = u_energy;
                    }
                }
            }
            //for(int k=0; k<W*H; k++ ){
            //	// Map the color to a label
            //	int c = getColor(im + 3 * k);
            //       int i;
            //	for(i=0;i<nColors && c!=colors[i]; i++ );
            //	if (c && i==nColors){
            //		if (i<M)

            //               colors[nColors++] = c;
            //		else
            //			c=0;
            //	}

            //   // Set the energy
            //   float* r = res + k * M;
            //	if (c){
            //		for(int j=0; j<M; j++ )
            //			r[j] = n_energy;
            //		r[i] = p_energy;
            //	}
            //	else{
            //		for(int j=0; j<M; j++ )
            //			r[j] = u_energy;
            //	}
            //}
            return res;

        }
           
    }
}
