import NeuralNetwork.*;
import GeneticAlgorithm.*;

NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;

void setup(){
  
  model = new NN_Model(this);
  in = new InputLayer(this, 36);
  lay1 = new HiddenLayer(this, 18, in, "relu");
  lay2 = new HiddenLayer(this, 18, lay1, "relu");
  out = new OutputLayer(this, 2, lay2, "sigmoid");
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  
  model.setLoss("mse");
  
  model.printParams();
  
  model.forward_prop();
  float[] y_ = {0.1, 0.9};
  float loss = model.compute_loss(y_);
  print("loss: ");
  print(loss);
  
}

void draw(){
  
}
