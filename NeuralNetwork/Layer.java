package NeuralNetwork;

import java.lang.Math;
import processing.core.*;
import processing.core.PApplet;

public class Layer{
  // The parent Processing applet
  protected final PApplet parent;
    
  float[][] weights;
  public float [] neurons;
  public int nNeurons;
  public Layer prevLayer;
  public String lay_type;
  public String activ_type;
  int nParameters = 0;
  
  public Layer(PApplet parent, int nNeu, Layer prev_Layer, String activation_type, String layer_type){
    this.parent = parent;
    this.prevLayer = prev_Layer;
    this.lay_type = layer_type;
    this.activ_type = activation_type;
    this.nNeurons = nNeu;
    weights = new float[nNeu][prevLayer.nNeurons+1];
    //weights = new float[nNeu][prevLayer.nNeurons];
    neurons = new float [nNeu];
    init();
    nParameters = weights.length * weights[0].length;
  }
  
  public Layer(PApplet parent, int nNeu, String layer_type){
    this.parent = parent;
    this.nNeurons = nNeu;
    this.lay_type = layer_type;
    neurons = new float [nNeu];
  }
  
  public void init(){
    for (int i = 0; i < nNeurons; i++){
      for (int j = 0; j < prevLayer.nNeurons+1; j++){
      //for (int j = 0; j < prevLayer.nNeurons; j++){
        weights[i][j] = parent.random((float)-1.0,(float) 1.0);
      }
    }
  }
  
  public void compute_output(){
    //inputs x Weights (+ bias)
    for (int i = 0; i < nNeurons; i++){
      float sum = 0;
      /*for (int j = 0; j < prevLayer.nNeurons; j++){
        sum += weights[i][j] * prevLayer.neurons[j];
      }*/
      for (int j = 0; j < prevLayer.nNeurons+1; j++){
        if(j == prevLayer.nNeurons){
          sum += weights[i][j];
        }
        else{
          sum += weights[i][j] * prevLayer.neurons[j];
        } 
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
    else if (activ_type == "tanh"){
      neurons = tanh(neurons, nNeurons);
    }
    else{
      parent.println("ERROR: No activation funcion selected");
    }
  }
  
  public void setWeights(float [][] w){
    weights = w;
  }
  
  public void printParams(){
    parent.println(lay_type);
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

  public float [] tanh (float z[],int n){
    float [] s = new float [n];
    for (int i = 0; i < n; i++){
      s[i] = (float)Math.tanh(z[i]);
    }
    return s;
  }
}
