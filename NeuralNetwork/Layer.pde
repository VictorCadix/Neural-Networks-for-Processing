class Layer{
  float[][] weights;
  int nNeurons;
  Layer prevLayer;
  
  Layer(int nNeurons, Layer prev_Layer){
    prevLayer = prev_Layer;
    weights = new float[nNeurons][prevLayer.nNeurons];
    init();
  }
  
  void init(){
    for (int i = 0; i < nNeurons; i++){
      for (int j = 0; j < prevLayer.nNeurons; j++){
        weights[i][j] = random(-1.0, 1.0);
      }
    }
  }
}

class InputLayer() extends Layer{
  InputLayer(){
    super();
  }
}
