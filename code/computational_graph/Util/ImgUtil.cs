using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DenseCRF
{
  public  class ImgUtil
    {
        /// <summary>
        /// 双线性插值
        /// </summary>
        /// <param name="array">二维数组</param>
        /// <param name="length_0">输出的宽</param>
        /// <param name="length_1">输出的高</param>
        /// <returns></returns>
        public static double[,] BilinearInterp(double[,] array, int length_0, int length_1)
        {
            double[,] _out = new double[length_0, length_1];
            int original_0 = array.GetLength(0);
            int original_1 = array.GetLength(1);

            float ReScale_0 = original_0 / ((float)length_0);  // 倍数的倒数
            float ReScale_1 = original_1 / ((float)length_1);

            float index_0;
            float index_1;
            int inde_0;
            int inde_1;
            float s_leftUp;
            float s_rightUp;
            float s_rightDown;
            float s_leftDown;

            for (int i = 0; i < length_0; i++)
            {
                for (int j = 0; j < length_1; j++)
                {
                    index_0 = i * ReScale_0;
                    index_1 = j * ReScale_1;
                    inde_0 = (int)Math.Floor(index_0);
                    inde_1 = (int)Math.Floor(index_1);
                    s_leftUp = (index_0 - inde_0) * (index_1 - inde_1);
                    s_rightUp = (inde_0 + 1 - index_0) * (index_1 - inde_1);
                    s_rightDown = (inde_0 + 1 - index_0) * (inde_1 + 1 - index_1);
                    s_leftDown = (index_0 - inde_0) * (inde_1 + 1 - index_1);
                    _out[i, j] = array[inde_0, inde_1] * s_rightDown + array[inde_0 + 1, inde_1] * s_leftDown + array[inde_0 + 1, inde_1 + 1] * s_leftUp + array[inde_0, inde_1 + 1] * s_rightUp;
                }
            }

            return _out;
        }
        public static float[,] BilinearInterp(float[,] array, int length_0, int length_1)
        {
            float[,] _out = new float[length_0, length_1];
            int original_0 = array.GetLength(0);
            int original_1 = array.GetLength(1);

            float ReScale_0 = original_0 / ((float)length_0);  // 倍数的倒数
            float ReScale_1 = original_1 / ((float)length_1);

            float index_0;
            float index_1;
            int inde_0;
            int inde_1;
            float s_leftUp;
            float s_rightUp;
            float s_rightDown;
            float s_leftDown;

            for (int i = 0; i < length_0; i++)
            {
                for (int j = 0; j < length_1; j++)
                {
                    index_0 = i * ReScale_0;
                    index_1 = j * ReScale_1;
                    inde_0 = (int)Math.Floor(index_0);
                    inde_1 = (int)Math.Floor(index_1);
                    s_leftUp = (index_0 - inde_0) * (index_1 - inde_1);
                    s_rightUp = (inde_0 + 1 - index_0) * (index_1 - inde_1);
                    s_rightDown = (inde_0 + 1 - index_0) * (inde_1 + 1 - index_1);
                    s_leftDown = (index_0 - inde_0) * (inde_1 + 1 - index_1);
                    _out[i, j] = array[inde_0, inde_1] * s_rightDown + array[inde_0 + 1, inde_1] * s_leftDown + array[inde_0 + 1, inde_1 + 1] * s_leftUp + array[inde_0, inde_1 + 1] * s_rightUp;
                }
            }

            return _out;
        }
        public static float[,] Bilinear(float[,] data, int H1, int W1)
        {
            int nW = data.GetLength(0); int nH = data.GetLength(1);
            float fw = (float)(nW) / W1;
            float fh = (float)(nH) / H1;
            float[,] data2 = new float[W1, H1];
            //int y1, y2, x1, x2, x0, y0;
            //float fx1, fx2, fy1, fy2;
            for (var x = 0; x < W1; x = x + 1)
            {

                for (var y = 0; y < H1; y = y + 1)
                {

                    var x0 = (float)x * fw;
                    var y0 = (float)y * fh;
                    var x1 = (x0);
                    var x2 = x1 + 1;
                    var y1 = (y0);
                    var y2 = y1 + 1;
                    var fx1 = x0 - x1;
                    var fx2 = 1.0 - fx1;
                    var fy1 = y0 - y1;
                    var fy2 = 1.0 - fy1;

                    var s1 = fx1 * fy1;
                    var s2 = fx2 * fy1;
                    var s3 = fx2 * fy2;
                    var s4 = fx1 * fy2;
                    x2 = x2 >= nW ? nW - 1 : x2;
                    x1 = x1 >= nW ? nW - 1 : x1;
                    y2 = y2 >= nH ? nH - 1 : y2;
                    y1 = y1 >= nH ? nH - 1 : y1;

                    try
                    {
                        var DSS1 = data[(int)x2, (int)y2];
                        var DSS2 = data[(int)x1, (int)y2];
                        var DSS3 = data[(int)x1, (int)y1];
                        var DSS4 = data[(int)x2, (int)y1];
                        var uu =(float)DSS1 * s1 + DSS2 * s2 + DSS3 * s3 + DSS4 * s4;

                        data2[x, y] = (float)uu;
                    }
                    catch
                    {
                        //var ggg = 111;
                    }

                    // var DS = GetDFromUV(uu, vv);

                }

            }
            return data2;

        }
        public static void savefile2(float[,] dbz, string file)
        {
            System.Drawing.Bitmap bt1 = new System.Drawing.Bitmap(dbz.GetLength(0), dbz.GetLength(1));
            for (int c = 0; c < dbz.GetLength(0); c++)
            {
                for (int j = 0; j < dbz.GetLength(1); j++)
                {
                    double daya = (dbz[c, j]) * 70;

                    #region 判断色标
                    if (daya > 0)
                    {
                        bt1.SetPixel(c, j, getldcolor(daya));
                    }

                    #endregion
                }
            }
            bt1.Save(file);
        }
        public static void savefile(float[,] dbz, string file)
        {
            System.Drawing.Bitmap bt1 = new System.Drawing.Bitmap(dbz.GetLength(0), dbz.GetLength(1));
            for (int c = 0; c < dbz.GetLength(0); c++)
            {
                for (int j = 0; j < dbz.GetLength(1); j++)
                {
                    double daya = (dbz[c, j])*100;

                    #region 判断色标
                    if (daya > 0)
                    {
                        bt1.SetPixel(c, j, getldcolor(daya));
                    }

                    #endregion
                }
            }
            bt1.Save(file);
        }
        static Color getldcolor(double daya)
        {
            if (daya <= -5)
            {
                return Color.FromArgb(0, 0, 0, 0);
                //return Color.FromArgb(0x00, 0xAC, 0xA4);
            }
            else if (daya <= 0)
            {
                return Color.FromArgb(0, 0, 0, 0);
                // return Color.FromArgb(0xC0, 0xC0, 0xFE);
            }
            else if (daya <= 5)
            {
                return Color.FromArgb(0, 0, 0, 0);
            }
            else if (daya <= 10)
            {
                return Color.FromArgb(0x1E, 0x1E, 0xD0);
            }
            else if (daya <= 15)
            {
                return Color.FromArgb(0xA6, 0xFC, 0xA8);
            }
            else if (daya <= 20)
            {
                return Color.FromArgb(0x00, 0xEA, 0x00);
            }
            else if (daya <= 25)
            {
                return Color.FromArgb(0x10, 0x92, 0x1A);
            }
            else if (daya <= 30)
            {
                return Color.FromArgb(0xFC, 0xF4, 0x64);
            }
            else if (daya <= 35)
            {
                return Color.FromArgb(0xC8, 0xC8, 0x02);
            }
            else if (daya <= 40)
            {
                return Color.FromArgb(0x8C, 0x8C, 0x00);

            }
            else if (daya <= 45)
            {
                return Color.FromArgb(0xFE, 0xAC, 0xAC);

            }
            else if (daya <= 50)
            {
                return Color.FromArgb(0xFE, 0x64, 0x5C);

            }
            else if (daya <= 55)
            {
                return Color.FromArgb(0xEE, 0x02, 0x30);

            }
            else if (daya <= 60)
            {
                return Color.FromArgb(0xD4, 0x8E, 0xFE);

            }
            else if (daya <= 65 || daya >= 65)
            {
                return Color.FromArgb(0xAA, 0x24, 0xFA);

            }
            return Color.FromArgb(0, 0, 0, 0);
        }
    }
}
