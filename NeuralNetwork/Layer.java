package NeuralNetwork;

import processing.core.*;
import processing.core.PApplet;

public class Layer{
  // The parent Processing applet
  protected final PApplet parent;
    
  float[][] weights;
  float [] neurons;
  int nNeurons;
  public Layer prevLayer;
  public String layer_type;
  public String activ_type;
  int nParameters = 0;
  
  public Layer(PApplet parent, int nNeurons, Layer prev_Layer, String activation_type){
    this.parent = parent;
    prevLayer = prev_Layer;
    this.nNeurons = nNeurons;
    weights = new float[nNeurons][prevLayer.nNeurons];
    neurons = new float [nNeurons];
    layer_type = "";
    activ_type = activation_type;
    init();
    nParameters = weights.length * weights[0].length;
  }
  
  public Layer(PApplet parent, int nNeurons){
    this.parent = parent;
    this.nNeurons = nNeurons;
    neurons = new float [nNeurons];
    layer_type = "";
  }
  
  public void init(){
    for (int i = 0; i < nNeurons; i++){
      for (int j = 0; j < prevLayer.nNeurons; j++){
        weights[i][j] = parent.random((float)-1.0,(float) 1.0);
      }
    }
  }
  
  public void compute_output(){
    //inputs x Weights (+ bias)
    for (int i = 0; i < nNeurons; i++){
      float sum = 0;
      for (int j = 0; j < prevLayer.nNeurons; j++){
        sum += weights[i][j] * prevLayer.neurons[j];
      }
      neurons[i] = sum;
    }
  }
  
  public void activate(){
    if (activ_type == "relu"){
      neurons = relu(neurons, nNeurons);
    }
    else if (activ_type == "sigmoid"){
      neurons = sigmoide(neurons, nNeurons);
    }
    else if (activ_type == "softmax"){
      neurons = softmax(neurons, nNeurons);
    }
    else{
      parent.println("ERROR: No activation funcion selected");
    }
  }
  
  public void setWeights(float [][] w){
    weights = w;
  }
  
  public void printParams(){
    parent.println(layer_type);
    parent.print("\tNumber neurons: ");
    parent.println(neurons.length);
    parent.print("\tWeights dimension: (");
    try{
      parent.print(weights.length);
      parent.print(",");
      parent.print(weights[0].length);
      parent.println(")");
    }
    catch (NullPointerException e){
      parent.println("NULL)");
    }
    
    parent.print("\tActivation function: ");
    parent.println(activ_type);
    
     parent.print("\tNumber of parameters: ");
     parent.println(nParameters);
  }
  
  public float [] sigmoide (float z[],int n){
  float [] s = new float [n];
  for (int i = 0; i < n; i++){
    s[i] = 1/(1+ parent.exp(-z[i]));
  }
  return s;
  }
  
  public float [] relu (float z[],int n){
  float [] s = new float [n];
  for (int i = 0; i < n; i++){
    s[i] = parent.max(0,z[i]);
  }
  return s;
  }

  public float [] softmax (float z[],int n){
  float [] s = new float [n];
  float sum = 0;
  for (int i = 0; i < n; i++){
    sum += parent.exp(z[i]);
    if (sum > 1e38){
      sum = (float)1e38;
    }
  }
  for (int i = 0; i < n; i++){
    s[i] = parent.exp(z[i]) / sum;
  }
  return s;
  }
}

public class InputLayer extends Layer{
  public InputLayer(PApplet parent,int nNeurons){
    super(parent, nNeurons);
    layer_type = "input_layer";
  }
  
  public void setNeurons(float []num){
    this.neurons = num;
  }
}

public class OutputLayer extends Layer{
  public OutputLayer(PApplet parent, int nNeurons, Layer prev_Layer, String activation_type){
    super(parent, nNeurons, prev_Layer, activation_type);
    layer_type = "output_layer";
  }
}

public class HiddenLayer extends Layer{
  public HiddenLayer(PApplet parent, int nNeurons, Layer prev_Layer, String activation_type){
    super(parent, nNeurons, prev_Layer, activation_type);
    layer_type = "hidden_layer";
  }
  
  public int num_correct(){
    int index = 0;
    
    for (int i = 1; i < nNeurons; i++){
      if (neurons[i] > neurons[index]){
        index = i;
      }
    }
    return index;
  }
  
  public float prob_numcorrect(){
    int index = 0;
    
    for (int i = 1; i < nNeurons; i++){
      if (neurons[i] > neurons[index]){
        index = i;
      }
    }
    return neurons[index];
  }
}
