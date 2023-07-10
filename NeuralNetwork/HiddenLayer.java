package NeuralNetwork;

import processing.core.*;
import processing.core.PApplet;

public class HiddenLayer extends Layer{
  public HiddenLayer(PApplet parent, int nNeurons, Layer prev_Layer, String activation_type){
    super(parent, nNeurons, prev_Layer, activation_type, "hidden_layer");
  }
}
