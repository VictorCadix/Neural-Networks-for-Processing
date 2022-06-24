class NN_Model{
  ArrayList<Layer> layers;
  
  NN_Model(){
    layers = new ArrayList<Layer>();
  }
  
  void addLayer(Layer new_layer){
    layers.add(new_layer);
  }
  
  void printParams(){
    print("number of layers: ");
    println(layers.size());
    
    int i = 0;
    for(Layer layer : layers){
      print("layer " + str(i) + " -> ");
      layer.printParams();
      i++;
    }
  }
}
