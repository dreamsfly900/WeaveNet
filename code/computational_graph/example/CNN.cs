using CNN;
using computational_graph.Layer;
using computational_graph.loss;
using DenseCRF;
using FCN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace computational_graph.example
{
    public class CNNtest
    {
        static void Main(string[] args)
        {
            Minst.MinstImgArr trainImg = Minst.read_Img("D:\\caffe\\Minst\\train-images.idx3-ubyte");
            Minst.MinstLabelArr trainLabel = Minst.read_Lable("D:\\caffe\\Minst\\train-labels.idx1-ubyte");
            float[][][,] anno = new float[1][][,];
            List<float[][][,]> annolist = new List<float[][][,]>();
            List<float[][]> taglist = new List<float[][]>();
            float[][][,] annolisttest = new float[1][][,]; 
            float[][] output = new float[1][];

            int sss = 0;
            for (int a = 0; a < 30000; a++)
            {
                output = new float[1][];
                anno = new float[1][][,];
                anno[0] = new float[1][,];
                anno[0][0] = trainImg.ImgPtr[a].ImgData;
                output[0] = trainLabel.LabelPtr[a].LabelData;
                annolist.Add(anno);
                taglist.Add(output);
                if (a == sss)
                    annolisttest = (anno);
            }
          //  MSELoss mSELoss = new MSELoss();
            cross_entropy mSELoss = new cross_entropy();
            CNN cNN = new CNN();
            int count = 0;
            while (count < 1)
            {
                for (int a = 0; a < 30000; a++)
                {

                    dynamic data = cNN.Forward(annolist[a]);
                    var loss = mSELoss.Forward(data, taglist[a]);
                    
                    Console.WriteLine("误差:" + loss);
                    dynamic Ddata = mSELoss.Backward();
                    cNN.backward(Ddata);
                    cNN.update();
                }
                count++;
            }
             
            Minst.MinstImgArr testImg = Minst.read_Img("D:\\caffe\\Minst\\t10k-images.idx3-ubyte");
            Minst.MinstLabelArr testLabel = Minst.read_Lable("D:\\caffe\\Minst\\t10k-labels.idx1-ubyte");
            var surss = 0;
            var shibai=0;
            for (int a = 0; a < 1500; a++)
            {

                float[][][,] annotest = new float[1][][,];
                annotest[0] = new float[1][,];
                annotest[0][0] = testImg.ImgPtr[a].ImgData;
                float[][] outputtest = new float[1][];
                outputtest = new float[1][];
                outputtest[0] = testLabel.LabelPtr[a].LabelData;
                dynamic fenlei = cNN.Forward(annotest);

                if (vecmaxIndex(fenlei, 10) == vecmaxIndex(outputtest, 10))
                {
                    surss++;


                } else
                    shibai++;
                   

                Console.WriteLine("成功：" + surss+ "失败："+ shibai);
            }
            Console.ReadLine();
        }
        static int vecmaxIndex(float[][] vec, int veclength)// 返回向量最大数的序号
        {
            int i;
            float maxnum = -1.0f;
            int maxIndex = 0;
            for (i = 0; i < vec.Length; i++)
            {
                for (int j = 0; j < veclength; j++)
                    if (maxnum < vec[i][j])
                    {
                        maxnum = vec[i][j];
                        maxIndex = j;
                    }
            }
            return maxIndex;
        }
    }
   public class CNN
    {
        Conv2DLayer cl;
        
      
       
        Conv2DLayer cl2;
        Conv2DLayer cl3;
        //TanhLayer sl = new TanhLayer();
        //TanhLayer sl2 = new TanhLayer();
        //TanhLayer sl3 = new TanhLayer();
        Maxpooling ap1;
        Maxpooling ap2;
        SigmodLayer sl = new SigmodLayer();
        SigmodLayer sl2 = new SigmodLayer();
        //SigmodLayer sl3 = new SigmodLayer();

        Softmax sl3 = new Softmax();
        //Averpooling ap2;
        //Averpooling ap1;



        public CNN()
        {
              cl = new Conv2DLayer(1, 0, 5, 1, 6);
              //ap1 = new Averpooling(2);
            ap1 = new Maxpooling(2);
            cl2 = new Conv2DLayer(1, 0, 5, 6, 12);
            // ap2 = new Averpooling(2);
            ap2 = new Maxpooling(2);
              cl3 = new Conv2DLayer(innum:12,outnum:10, _inSize: 4,_full:true );
        }
        public dynamic Forward(float[][][,] matrices)
        {
            dynamic data = cl.Forward(matrices);
            data = sl.Forward(data);
            data = ap1.Forward(data);
            data = cl2.Forward(data);
            data = sl2.Forward(data);
            data = ap2.Forward(data);
            data = cl3.Forward(data);
            data = sl3.Forward(data);
            return data;
        }
        dynamic cl3grid;
        dynamic cl2grid;
        dynamic clgrid;
        public void backward(dynamic grid)
        {

            dynamic grid2 = sl3.Backward(grid);

            cl3grid = cl3.backweight(grid2);//获取cl3的权重

            //--------------------------------
              

            grid2 = cl3.backward(grid2);
            grid2 =ap2.backward(grid2);
            grid2 = sl2.Backward(grid2);

            cl2grid = cl2.backweight(grid2);//获取cl2的权重
            //-------------------------------------

            grid2 = cl2.backward(grid2);
            grid2 = ap1.backward(grid2);
            grid2 = sl.Backward(grid2);

            clgrid = cl.backweight(grid2);//获取cl的权重


        }
        float lr = 1.0f;
        public void update()
        {
            
        //    int channl = cl3grid.grid.Length;

            cl3.wdata = Matrix.MatrixSub(cl3.wdata, Matrix.multiply(cl3grid.grid, lr)); 
            cl3.basicData = Matrix.MatrixSub(cl3.basicData, Matrix.multiply(cl3grid.basic, lr));

            cl2.weights = Matrix.MatrixSub(cl2.weights, Matrix.multiply(cl2grid.grid, lr));
            cl2.basicData = Matrix.MatrixSub(cl2.basicData, Matrix.multiply(cl2grid.basic, lr));

            cl.weights = Matrix.MatrixSub(cl.weights, Matrix.multiply(clgrid.grid, lr));
            cl.basicData = Matrix.MatrixSub(cl.basicData, Matrix.multiply(clgrid.basic, lr));
        }
    }
}
