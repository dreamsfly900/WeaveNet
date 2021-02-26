using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
 public   class Class1
    {
	public static	void convolution_3D_shared( float[] in1, float[] mask, float[] out1 , int maskwidth, int w, int h, int outw, int outh, int stride, int p, int outChannels,  int inChannels)
		 {


		
		int inChannel =0;
		int outChannel = 0;
		int outchannelnum = outChannel * w * h;
		int inks = (inChannel) * maskwidth * maskwidth;
		int inchannelnumk = inChannels * maskwidth * maskwidth;
		int inchannelnum = inChannel * w * h;
		int inko = inChannel * outw * outh;
		int inchannelnumo = inChannels * outw * outh;
			 /* int i = outChannel;
			  int j = inChannel;*/
		 	//extern __shared__ float[];
			/* __shared__ float  subA[bs][bs];
			 __shared__ float  subB[bs][bs];*/
			 if (outChannel<outChannels && inChannel<inChannels)
			 {
				// convolution();
			/*	 out[inChannel] += outChannel+inChannel;
				 return;*/
				 // in[channelnum];
				 int x = w;
		int y = h;
		int x2 = outw;
		int y2 = outh;
		int row = ((x - x2) + 2 * p) / stride + 1;
		int col = ((y - y2) + 2 * p) / stride + 1;
		//  float[, ] temp = new float[row, col];
		int aaa = 0;
		int nx = 0;
		int pp = 0;
				//  outa[0] += 1;
				 for (int i = 0 - p; i<row; i = i + stride)
				 {
					 int ny = 0;
					 if (nx<row) {
						 ny = 0;
						 for (int j = 0 - p; j<col; j = j + stride)
						 {
							 if (ny<col) {
								 //var sum = 0.0f;
								 for (int i2 = 0; i2<x2; i2++)
									 for (int j2 = 0; j2<y2; j2++)
									 {
										 if (i + i2< 0 || j + j2< 0 || i + i2 >= x || j + j2 >= y)
										 {
											 continue;
										 }
										 else {
											// int a = in[(  inchannelnum) + (i + ((j + i2) * w) + j2)];
											 
											 
											    // out[inChannel] = in[inchannelnum+(i + ((j + i2) * w) + j2)];
											 
											/* out[(outChannel * inchannelnumo) + inko + (nx * outw + ny)] 
												 += in[(inChannel*inchannelnum) + (i + ((j + i2) * w) + j2)] 
												 * mask[(outChannel * inchannelnumk) + inks + (i2 * maskwidth + j2)];*/
											 out1[(outChannel * inchannelnumo) + inko + (nx * outw + ny)]+=
											 in1[inchannelnum + (j + ((i + i2) * w) + j2)] * mask[(outChannel * inchannelnumk) + inks + (i2 * maskwidth + j2)];
	/* out[(outChannel* inchannelnumo)+ inko + (nx * outw + ny)]
		 += in[inchannelnum + (j + ((i + i2) * w) + j2)] * mask[(outChannel* inchannelnumk)+ inks +(i2 * maskwidth + j2)];*/
	//  out[] += in[channelnum]
}
										 //temp[nx, ny] += matrix[i + i2, j + j2] * kernel[i2, j2];

									 }
								 
								 ny++;
							 }
							 //temp[nx, ny] = Math.Max(ReLU, temp[nx, ny] + bias);
							
						 }
						 
						 nx++;
					 }
					
				 }
				// __syncthreads();
				
				 //if (tx==0)
				 //{
					// for (int inlen = 0; inlen < (lens); inlen++)
					// {
					//	 out[inlen] = outa[inlen];
					// }
					// //for (int inlen = 0; inlen < (inChannels); inlen++)
					// //{
					//	// int len = (row * col) * outChannels;
					//	// int index = inlen * (row * col) * outChannels;
					//	// for (int s = index; s < len; s++)
					//	// {
					//	//	 int r = s / row;
					//	//	 int c = (s+1) % row;
					//	//	 for (int a = 0; a < row; a++)
					//	//	 {
					//	//		 for (int b = 0; b < col; a++)
					//	//		 {
					//	//			 //outa[(s*(row * col)）]
					//	//		 }
					//	//	 }
					//	// }
					// //}
				 //}
				 
			 }
			 
		 }

    }
}
