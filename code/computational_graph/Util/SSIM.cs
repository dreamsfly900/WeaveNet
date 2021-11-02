using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.Util
{
   public class SSIM
    {
        int window_size = 11; bool size_average = true;
        int channel = 1;
        float[][][,] window;
        public SSIM(int window_size = 11,bool size_average = true,int channel=1)
        {
            this.window_size = window_size;
            this.size_average = size_average;
            this.channel = channel;
            this.window = create_window(window_size, this.channel);
        }
        public float[] gaussian(int window_size, float sigma)
        {
            var gauss = new float[window_size];
            for (int x = 0; x < window_size; x++)
            {
                gauss[x] = (float)Math .Exp(-(float)Math.Pow((x - window_size / 2),2) / (float)(2 * Math.Pow( sigma, 2)));
            }
            return Matrix.divide(gauss, Matrix.sum(gauss));
        }
        public float[][][,] create_window(int window_size, int channel)
        {
            var _1D_window = gaussian(window_size, 1.5f);
            var _1dw = Matrix.zroe2D(channel, 1, window_size, window_size);
            for (int i = 0; i < channel; i++) {
                for (int y = 0; y < window_size; y++)
                {
                    for (int x = 0; x < window_size; x++)
                    {
                        float[] p= Matrix.multiply(_1D_window, _1D_window[x]);
                       
                            for (int y2 = 0; y2 < window_size; y2++)
                                _1dw[i][0][x, y2] = p[y2];
                        
                    }
                } }
            // var _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            // window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

            return _1dw;
        }
        public float ssim(float[,] img1, float[,] img2)
        {
            return ssim(img1, img2, window[0][0], window_size, channel, size_average);
        }
        public float ssim(float[,] img1, float[,] img2, float[,] window, int window_size = 11, int channel = 1, bool size_average = true)
        {
            float[,] mu1 = Matrix.convnFull(img1, window, 1, window_size / 2);

            float[,] mu2 = Matrix.convnFull(img2, window, 1, window_size / 2);

            var mu1_sq = Matrix.Pow(mu1, 2);
            var mu2_sq = Matrix.Pow(mu2, 2);
            var mu1_mu2 = Matrix.multiply(mu1, mu2);
            var img1_1=  Matrix.multiply(img1, img1);
            var sigma1_sq= Matrix.MatrixSub(Matrix.convnFull(img1_1, window, 1, window_size / 2), mu1_sq);
            var img2_1 = Matrix.multiply(img2, img2);
            var sigma2_sq = Matrix.MatrixSub(Matrix.convnFull(img2_1, window, 1, window_size / 2), mu2_sq);
            var img1_2 = Matrix.multiply(img1, img2);
            var sigma12 = Matrix.MatrixSub( Matrix.convnFull(img1_2, window, 1, window_size / 2), mu1_mu2);
            var C1 = Math.Pow(0.01, 2);
            var C2 = Math.Pow(0.03, 2);
            // ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            var mu12=  Matrix.MAdd(Matrix.multiply(mu1_mu2, 2),(float)C1);
            var mu12sq = Matrix.MAdd(Matrix.multiply(sigma12, 2), (float)C2);
            var mu12sq_12= Matrix.multiply(mu12, mu12sq);

            var musq12=  Matrix.MAdd(Matrix.MatrixAdd(mu1_sq, mu2_sq).values, (float)C1);
            var mu12sq12 = Matrix.MAdd(Matrix.MatrixAdd(sigma1_sq, sigma2_sq).values, (float)C2);
            var mu12sq_12aq = Matrix.multiply(musq12, mu12sq12);
            var ssim_map=  Matrix.divide(mu12sq_12 ,mu12sq_12aq);
            if (this.size_average)
                return Matrix.Mean(ssim_map);
            return 0;
        }

    }
}
