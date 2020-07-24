using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
	public class Minst
	{

		public struct MinstImg
		{
			public int c;           // 图像宽
			public int r;           // 图像高
			public float[,] ImgData; // 图像数据二维动态数组
		}


		public struct MinstImgArr
		{
			public int ImgNum;        // 存储图像的数目
			public MinstImg[] ImgPtr;  // 存储图像数组指针
		}       // 存储图像数据的数组

		public struct MinstLabel
		{
			public int l;            // 输出标记的长
			public float[] LabelData; // 输出标记数据
		}

		public struct MinstLabelArr
		{
			public int LabelNum;
			public MinstLabel[] LabelPtr;
		}         // 存储图像标记的数组

	static	int ReverseInt(int i)
		{
			 int ch1, ch2, ch3, ch4;
			ch1 = i & 255;
			ch2 = (i >> 8) & 255;
			ch3 = (i >> 16) & 255;
			ch4 = (i >> 24) & 255;
			return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
		}

		public static MinstImgArr read_Img(string filename) // 读入图像
		{

			FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);
			BinaryReader br = new BinaryReader(fs);
			//cha = br.ReadChar();
			//num = br.ReadInt32();
			//doub = br.ReadDouble();
			//str = br.ReadString();
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			//从文件中读取sizeof(magic_number) 个字符到 &magic_number  
			magic_number = ReverseInt(br.ReadInt32());
			number_of_images = ReverseInt( br.ReadInt32());
			n_rows = ReverseInt(br.ReadInt32());
			n_cols = ReverseInt(br.ReadInt32());


			int i, r, c;

			// 图像数组的初始化
			MinstImgArr imgarr = new MinstImgArr();
			imgarr.ImgNum = number_of_images;
			imgarr.ImgPtr = new MinstImg[number_of_images];

			for (i = 0; i < number_of_images; ++i)
			{
				imgarr.ImgPtr[i].r = n_rows;
				imgarr.ImgPtr[i].c = n_cols;
				imgarr.ImgPtr[i].ImgData = new float[n_rows, n_cols];
				for (r = 0; r < n_rows; ++r)
				{
					 
					for (c = 0; c < n_cols; ++c)
					{
						 int temp = 0;
						 temp=br.ReadByte();
						imgarr.ImgPtr[i].ImgData[r,c] = (float)temp / 255.0f;
					}
				}
			}
			br.Dispose();
			fs.Close();
			return imgarr;
		}
		public static MinstLabelArr  read_Lable(string filename)// 读入图像
{
			FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);
			BinaryReader br = new BinaryReader(fs);

			int magic_number = 0;
	     	int number_of_labels = 0;
	    	int label_long = 10;

			//从文件中读取sizeof(magic_number) 个字符到 &magic_number  
			 
		 
			//从文件中读取sizeof(magic_number) 个字符到 &magic_number  
			magic_number = ReverseInt(br.ReadInt32());
			number_of_labels = ReverseInt(br.ReadInt32());

			int i, l;

			// 图像标记数组的初始化
			MinstLabelArr labarr = new MinstLabelArr();
		labarr.LabelNum=number_of_labels;
	labarr.LabelPtr=new MinstLabel[number_of_labels];

	for(i = 0; i<number_of_labels; ++i)  
	{  
		labarr.LabelPtr[i].l=10;
				labarr.LabelPtr[i].LabelData = new float[label_long];
				int temp = 0;
				temp = br.ReadByte();
				labarr.LabelPtr[i].LabelData[(int)temp]=1.0f;    
	}

			br.Dispose();
			fs.Close();
			return labarr;	
}
	}
}
