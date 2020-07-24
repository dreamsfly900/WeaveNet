namespace computational_graph.Layer
{
  public  interface  Layer
    {
         dynamic Forward(dynamic x);
        dynamic Backward(dynamic dout);
    }
}