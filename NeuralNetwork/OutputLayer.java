package NeuralNetwork;

import processing.core.*;
import processing.core.PApplet;

public class OutputLayer extends Layer{
  public OutputLayer(PApplet parent, int nNeurons, Layer prev_Layer, String activation_type){
    super(parent, nNeurons, prev_Layer, activation_type);
    this.layer_type = "output_layer";
  }
}
