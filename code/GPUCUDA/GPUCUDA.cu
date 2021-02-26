#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <iostream>
//#include<stdio.h>
using namespace std;
//返回thread和block
int getThreadNum()
{
	cudaDeviceProp prop;//cudaDeviceProp的一个对象
	int count = 0;//GPU的个数
	cudaGetDeviceCount(&count);
	std::cout << "gpu 的个数：" << count << '\n';

	cudaGetDeviceProperties(&prop, 0);//第二参数为那个gpu
	cout << "最大线程数：" << prop.maxThreadsPerBlock << endl;
	cout << "最大网格类型：" << prop.maxGridSize[0] << '\t' << prop.maxGridSize[1] << '\t' << prop.maxGridSize[2] << endl;
	return prop.maxThreadsPerBlock;
}
__global__ void conv(float* imgGpu, float* kernelGpu, float* resultGpu, int width, int height, int kernelSize)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= width * height)
	{
		return;
	}
	int row = id / width;//获取img 的行和列
	int clo = id % width;
	//每一个线程处理一次卷积计算
	//resultGpu[id] = 0;
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			float imgValue = 0;//记录结果
			//imgValue += kernelGpu[i*kernelSize + j] * imgGpu[id];
			int curRow = row - kernelSize / 2 + i;
			int curClo = clo - kernelSize / 2 + j;
			if (curRow < 0 || curClo < 0 || curRow >= height || curClo >= width)
			{
			}
			else
			{
				//imgValue += kernelGpu[i*kernelSize + j] * imgGpu[(curRow + i - 1)*width + curClo + j - 1];
				imgValue = imgGpu[curRow * width + curClo];

			}
			resultGpu[id] += kernelGpu[i * kernelSize + j] * imgValue;

		}
	}
}

//形参：枚举类型
void GetCudaCalError(cudaError err)
{
	if (err != cudaSuccess)
	{
		cout << "分配内存失败！程序结束！";
	}
	return;
}
int main()
{
	//定义一个1080p照片
	const int width = 1920;
	const int height = 1080;
	//float *img = (float*)calloc(width*height, sizeof(float));
	float* img = new float[width * height];
	//赋值
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			img[col + row * width] = (col + row) % 256;
		}
	}
	//声明卷积核大小,大小为3*3
	const int kernelSize = 3;
	//float*kernel = (float*)calloc(kernelSize*kernelSize, sizeof(float));
	float* kernel = new float[kernelSize * kernelSize];
	//卷积核赋值
	//第一种方法
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			kernel[i + j * kernelSize] = i - 1;
		}
	}
	//第二种
	/*for (int i = 0; i < kernelSize*kernelSize; ++i)
	{
		kernel[i] = i % kernelSize - 1;
	}*/
	//输出img的左上角
	for (int row = 0; row < 10; ++row)
	{
		for (int col = 0; col < 10; ++col)
		{
			std::cout << img[col + row * width] << '\t';
		}
		std::cout << '\n';
	}
	cout << "kernel\n";
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			std::cout << kernel[i * kernelSize + j] << '\t';
		}
		cout << endl;

	}


	float* imgGpu = 0;//将host值复制到device上面
	float* kernelGpu = 0;//将kernel也复制到device上
	float* resultGpu = 0;//卷积结果

	//为Device分配内存
	GetCudaCalError(cudaMalloc(&imgGpu, height * width * sizeof(float)));
	GetCudaCalError(cudaMalloc(&kernelGpu, kernelSize * kernelSize * sizeof(float)));
	GetCudaCalError(cudaMalloc(&resultGpu, height * width * sizeof(float)));
	//这个地方捕捉错误，明天改

	cudaMemcpy(imgGpu, img, width * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernelGpu, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
	//获取GPU信息
	const int threadNum = getThreadNum();
	const int blockNum = (width * height + threadNum - 1) / threadNum;//这里block使用一维
	//conv(imgGpu, kernelGpu, resultGpu, width, height, kernelSize);
    conv <<<blockNum, threadNum >>> (imgGpu, kernelGpu, resultGpu, width, height, kernelSize);
	//接受Device上resultGpu里面的数据
	float* showImg = new float[height * width];
	cudaMemcpy(showImg, resultGpu, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	for (int row = 0; row < 10; ++row)
	{
		for (int col = 0; col < 10; ++col)
		{
			std::cout << showImg[col + row * width] << '\t';
		}
		std::cout << '\n';
	}
	//没有释放内存
	cudaFree(imgGpu);
	cudaFree(kernelGpu);
	cudaFree(resultGpu);
	/*free(img);
	free(kernel);*/
	delete[] img;
	delete[] kernel;
	delete[] showImg;
	system("pause");
	return 0;
}