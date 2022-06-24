class Layer{
  float[][] weights;
  float [] neurons;
  int nNeurons;
  Layer prevLayer;
  
  Layer(int nNeurons, Layer prev_Layer){
    prevLayer = prev_Layer;
    nNeurons=nNeurons;
    weights = new float[nNeurons][prevLayer.nNeurons];
    neurons = new float [nNeurons];
    neurons = activacion();
    init();
  }
  
  Layer(int nNeurons){
    nNeurons=nNeurons;
    neurons = new float [nNeurons];
     for (int i = 0; i < nNeurons; i++){
      neurons[i] = random(0.0,1.0);
    }
  }
  
  void init(){
    for (int i = 0; i < nNeurons; i++){
      for (int j = 0; j < prevLayer.nNeurons; j++){
        weights[i][j] = random(-1.0, 1.0);
      }
    }
  }
  
  float [] activacion(){
    float [] t = new float [nNeurons];
      for (int i = 0; i < nNeurons; i++){
        float sum = 0;
        for (int j = 0; j < prevLayer.nNeurons; j++){
          sum += weights[i][j] * prevLayer.neurons[j];
      }
          t[i] = sum;
      }
      
      t = sigmoide(t,nNeurons);
      return t;
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
  HiddenLayer(int nNeurons ,Layer prev_Layer){
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
