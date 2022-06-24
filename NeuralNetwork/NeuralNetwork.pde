NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;

void setup(){
  
  model = new NN_Model();
  in = new InputLayer(36);
  lay1 = new HiddenLayer(18,in);
  lay2 = new HiddenLayer(18,lay1);
  out = new OutputLayer(10,lay2);
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  
}

void draw(){
  
}
