package NeuralNetwork;

import processing.core.*;
import processing.core.PApplet;

public class InputLayer extends Layer{
  public InputLayer(PApplet parent,int nNeurons){
    super(parent, nNeurons);
    layer_type = "input_layer";
  }
  
  public void setNeurons(float []num){
    this.neurons = num;
  }
}
