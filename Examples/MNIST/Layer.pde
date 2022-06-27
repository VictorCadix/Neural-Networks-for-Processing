class Layer{
  float[][] weights;
  float [] neurons;
  int nNeurons;
  Layer prevLayer;
  String layer_type;
  String activ_type;
  
  Layer(int nNeurons, Layer prev_Layer, String activation_type){
    prevLayer = prev_Layer;
    this.nNeurons = nNeurons;
    weights = new float[nNeurons][prevLayer.nNeurons];
    neurons = new float [nNeurons];
    layer_type = "";
    activ_type = activation_type;
    init();
  }
  
  Layer(int nNeurons){
    this.nNeurons = nNeurons;
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
  
  void compute_output(){
    //inputs x Weights (+ bias)
    for (int i = 0; i < nNeurons; i++){
      float sum = 0;
      for (int j = 0; j < prevLayer.nNeurons; j++){
        sum += weights[i][j] * prevLayer.neurons[j];
      }
      neurons[i] = sum;
    }
  }
  
  void activate(){
    if (activ_type == "relu"){
      neurons = relu(neurons, nNeurons);
    }
    else if (activ_type == "sigmoid"){
      neurons = sigmoide(neurons, nNeurons);
    }
    else{
      println("ERROR: No activation funcion selected");
    }
  }
  
  void printParams(){
    println(layer_type);
    print("\tNumber neurons: ");
    println(neurons.length);
    print("\tWeights dimension: (");
    try{
      print(weights.length);
      print(",");
      print(weights[0].length);
      println(")");
    }
    catch (NullPointerException e){
      println("NULL)");
    }
    
    print("\tActivation function: ");
    println(activ_type);
    
    //if(layer_type == "output_layer"){
      //print("\tCost function: ");
      //OutputLayer out = (OutputLayer)this;
      //println(out.costo);
      //print("\tMSE: ");
      //println(out.error);
    //}
  }
}

class InputLayer extends Layer{
  InputLayer(int nNeurons){
    super(nNeurons);
    layer_type = "input_layer";
  }
}

class OutputLayer extends Layer{
  //float costo, error;
  
  OutputLayer(int nNeurons, Layer prev_Layer, String activation_type){
    super(nNeurons, prev_Layer, activation_type);
    layer_type = "output_layer";
    //costo = func_costo(neurons, nNeurons, 4);
    //error = mse(neurons, nNeurons, 4);
  }
}

class HiddenLayer extends Layer{
  HiddenLayer(int nNeurons, Layer prev_Layer, String activation_type){
    super(nNeurons, prev_Layer, activation_type);
    layer_type = "hidden_layer";
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
