package NeuralNetwork;

import processing.core.*;
import processing.core.PApplet;

public class OutputLayer extends Layer{
  public OutputLayer(PApplet parent, int nNeurons, Layer prev_Layer, String activation_type){
    super(parent, nNeurons, prev_Layer, activation_type, "output_layer");
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
