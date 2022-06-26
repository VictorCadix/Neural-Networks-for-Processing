class NN_Model{
  ArrayList<Layer> layers;
  
  NN_Model(){
    layers = new ArrayList<Layer>();
  }
  
  void addLayer(Layer new_layer){
    layers.add(new_layer);
  }
  
  void printParams(){
    print("number of layers: ");
    println(layers.size());
    
    int i = 0;
    for(Layer layer : layers){
      print("layer " + str(i) + " -> ");
      layer.printParams();
      i++;
    }
  }
  
  void forward_prop(){
    for (int i = 1; i < layers.size(); i++){
      Layer layer = layers.get(i);
      layer.compute_output();
      layer.activate();
    }
  }
  
  float compute_loss(float[] y_){
    int output_layer = layers.size() - 1;
    float[] estimated = layers.get(output_layer).neurons;
    float loss = mse(estimated, y_);
    return loss;
  }
}


//La capa de salida debe de calcular el costo de cada simulación. 
//Para ello, se hace el cuadrado del valor que se obtiene en cada 
//neurona de salida restando el valor que se desea obtener en cada 
//una de estas neuronas

float func_costo(float z[],int n,int want){
  float cost = 0;
  for (int i = 0; i < n; i++){
    if (n == want)
    cost += pow(z[i] - 1,2);
    else
    cost += pow(z[i] - 0,2);  
  }
  return cost;
}

float mse(float[] z, float[]y_){
  float mse = 0;
  int n = z.length;
  for (int i = 0; i < n; i++){
    mse += pow(z[i]-y_[i], 2);
  }
  return mse/n;
}

float mse(float z[],int n,int want){
  float mse = 0;
  for (int i = 0; i < n; i++){
    if (n == want)
    mse += pow(z[i]-1, 2);
    else
    mse += pow(z[i]-0, 2);  
  }
  return mse/n;
}
