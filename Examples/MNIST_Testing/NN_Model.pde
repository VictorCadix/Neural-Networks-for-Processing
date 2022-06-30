class NN_Model{
  ArrayList<Layer> layers;
  String loss_type;
  PrintWriter log_file, log_file2, log_file3;
  
  NN_Model(){
    layers = new ArrayList<Layer>();
    loss_type = "";
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
  
  void creatFiles(){
    //String name = "log_" + str(nIndiv) + "i_" + str(mutation_rate) + "m_" + str(nCrossPoints) + "cp_"  + str(elitism) + "E_#";
    String name = "LOSS";
    log_file = createWriter("Data/" + name + ".txt");
    log_file.println("Generation,best_fitness:");
    
    String name2 ="WEIGHTS";
    log_file2 = createWriter("Data/" + name2 + ".txt");
    log_file2.println("Weights best individual:");
  }
  
  void creatFilesTest(){
    String name = "TEST";
    log_file3 = createWriter("Data/" + name + ".txt");
    log_file3.println("Test MNIST:");
  }
  
  void saveParamsLoss(int generation, int best, Population pop){
    log_file.print(generation);
    log_file.print(":");
    log_file.println(pop.individuals[best].fitness);
  }
  
  void ParamsWeights(int best, Population pop){ 
    log_file2.print("{");
    for(int i= 0; i < pop.individuals[best].chromosome_length; i++){
    log_file2.print(String.format("%.8f", pop.individuals[best].chromosome[i]));
    //log_file2.print(weights_byte[i]);
      if(i < (pop.individuals[best].chromosome_length-1)){
        log_file2.print(" , ");
      }
    }
    log_file2.print(" }FINISH");
    exit1();
  }
  
  void testFiles(int seq, int num, float prob, int num_real){
    log_file3.println("Secuencia " + seq+ ": Numero red: " + num + "\t" + "Probabilidad: " + prob + "\t" + "Numero Real: " + num_real);
  }
  
  void exit1(){
    log_file.flush();
    log_file.close();
    log_file2.flush();
    log_file2.close();
    println("Archivo cerrado");
  }
  
  void exit2(){
    log_file3.flush();
    log_file3.close();
    println("Archivo cerrado");
  }

  void setLoss(String loss_type){
    this.loss_type = loss_type;
  }
  
  void forward_prop(){
    for (int i = 1; i < layers.size(); i++){
      Layer layer = layers.get(i);
      layer.compute_output();
      layer.activate();
    }
  }
  
  float compute_loss(float[] y_){
    int output_layer = layers.size() - 1;
    float[] estimated = layers.get(output_layer).neurons;
    if (loss_type == "mse"){
      float loss = mse(estimated, y_);
      return loss;
    }
    else if (loss_type == "mae"){
      float loss = mae(estimated, y_);
      return loss;
    }
    println("ERROR: Loss_type missmatch");
    return -1; 
  }
  
  void genes2weights(float[] chromosome){
  float [][] w1 = new float[neulay1][neuin];
  float [][] w2 = new float[neulay2][neulay1];
  float [][] w3 = new float[neuout][neulay2];
  int i = 0;
  if(i < (neuin*neulay1)){
    for (int j1 = 0; j1 < neulay1; j1++){
      for(int k1 = 0; k1 < neuin; k1++){
        w1 [j1][k1] = chromosome[i];
        i++;
      }
    }
  }
  if(i<(neuin*neulay1+neulay1*neulay2) && i>=(neuin*neulay1)){
    for (int j2 = 0; j2 < neulay2; j2++) {
       for(int k2 = 0; k2 < neulay1; k2++){
         w2 [j2][k2] = chromosome[i];
         i++;
       }
    }
  }
  if(i>=(neuin*neulay1+neulay1*neulay2) && i<nParameters){
    for (int j3 = 0; j3 < neuout; j3++) {
       for(int k3 = 0; k3 < neulay2; k3++){
         w3 [j3][k3] = chromosome[i];
         i++;
       }
    }
  }
  lay1.setWeights(w1);
  lay2.setWeights(w2);
  out.setWeights(w3);
  }
  
  void readWeights() {
  BufferedReader reader = createReader("C:\\Users\\david\\OneDrive\\Documentos\\GitHub\\Neural-Networks-for-Processing\\Examples\\MNIST\\Data\\WEIGHTS.txt");
  String line = null;
  try {
    while ((line = reader.readLine()) != null) {
      String [] w = split(line, TAB);
      float [] w_f = new float [w.length];
      /*for(int i = 0; i < w.length; i++){
        w_f[i] = float.valueOf(w[i]);
      }*/
      //println(Valueof(w));
      genes2weights(float(w));
    }
    reader.close();
  } catch (IOException e) {
    e.printStackTrace();
  }
  } 

}


//La capa de salida debe de calcular el costo de cada simulación. 
//Para ello, se hace el cuadrado del valor que se obtiene en cada 
//neurona de salida restando el valor que se desea obtener en cada 
//una de estas neuronas

float mae(float[] z, float[]y_){
  float mae = 0;
  int n = z.length;
  for (int i = 0; i < n; i++){
    mae += abs(z[i]-y_[i]);

  }
  return mae/n;
}

float mse(float[] z, float[]y_){
  float mse = 0;
  int n = z.length;
  for (int i = 0; i < n; i++){
    mse += pow(z[i]-y_[i], 2);
  }
  return mse/n;
}


float func_costo(float z[],int want){
  float costo = 0;
  int n = z.length;
  for (int i = 0; i < n; i++){
    if (n == want)
    costo += pow(z[i]-1, 2);
    else
    costo += pow(z[i]-0, 2);  
  }
  return costo;
}

public static byte[] floatToByteArray(float value) {
    int intBits =  Float.floatToIntBits(value);
    return new byte[] {
      (byte) (intBits >> 24), (byte) (intBits >> 16), (byte) (intBits >> 8), (byte) (intBits) };
}