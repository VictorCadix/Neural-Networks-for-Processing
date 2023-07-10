import NeuralNetwork.*;

NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;

void setup(){
  
  model = new NN_Model(this);
  in = new InputLayer(this, 8);
  lay1 = new HiddenLayer(this, 8, in, "tanh");
  lay2 = new HiddenLayer(this, 8, lay1, "tanh");
  out = new OutputLayer(this, 2, lay2, "tanh");
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  
  model.setLoss("mse");
  
  
  model.printParams();
  
  float [] entry = {1.0,0.5,0.25,0.2,1.0,0.5,0.25,0};
  in.setNeurons(entry);
  model.forward_prop();
  float[] y_ = {0.1, 0.9};
  float loss = model.compute_loss(y_);
  print("loss: ");
  print(loss);
  
}

void draw(){
  
}
