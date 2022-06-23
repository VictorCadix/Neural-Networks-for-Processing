class NN_Model{
  ArrayList<Layer> layers;
  
  NN_Model(){
    layers = new ArrayList<Layer>();
  }
  
  void addLayer(Layer new_layer){
    layers.add(new_layer);
  }
}
