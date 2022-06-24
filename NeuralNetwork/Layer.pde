class Layer{
  float[][] weights;
  int nNeurons;
  Layer prevLayer;
  
  Layer(int nNeurons, Layer prev_Layer){
    prevLayer = prev_Layer;
    nNeurons=nNeurons;
    weights = new float[nNeurons][prevLayer.nNeurons];
    init();
  }
  
  Layer(int nNeurons){
    nNeurons=nNeurons;
  }
  
  void init(){
    for (int i = 0; i < nNeurons; i++){
      for (int j = 0; j < prevLayer.nNeurons; j++){
        weights[i][j] = random(-1.0, 1.0);
      }
    }
  }
  
   
}

class InputLayer extends Layer{
  InputLayer(int nNeurons){
   super(nNeurons);
  }
}

class OutputLayer extends Layer{
  OutputLayer(int nNeurons, Layer prev_Layer){
    super(nNeurons, prev_Layer);
  }
}

class HiddenLayer extends Layer{
  HiddenLayer(int nNeurons, Layer prev_Layer){
    super(nNeurons, prev_Layer);
  }
}

float [] sigmoide (float z[],int n){
  float [] s = new float [n];
    for (int i = 0; i < n; i++){
        s[i] = 1/(1+ exp(-z[i]));
      }
   return s;
  }
  
float [] relu (float z[],int n){
  float [] s = new float [n];
    for (int i = 0; i < n; i++){
        s[i] = max(0,z[i]);
      }
   return s;
  }
