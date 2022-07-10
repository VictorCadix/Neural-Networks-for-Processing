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
    print("Total number of parameters: ");
    int nParams = 0;
    for(Layer layer : layers){
      nParams += layer.nParameters;
    }
    println(nParams);
    
    print("Loss function: ");
    println(loss_type);
    println("----------------------");
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
  
  void testFiles(int seq, int num, float prob, int num_real, Layer out){
    log_file3.println("Secuencia " + seq+ ": Numero red: " + num + "\t" + "Probabilidad: " + prob + "\t" + "Numero Real: " + num_real);
    log_file3.println("Neuronas de la capa de salida: ");
    int n = out.neurons.length;
    for (int i= 0; i < n; i++){
      log_file3.print(i+1 + ": " + out.neurons[i] + "\t");
      if(i == n-1){
        log_file3.println();
        log_file3.println();
      }
    }
  }
  
  void sucess(int nImg, int suc){
     log_file3.println("El exito de la red ha sido de: " + (((float)suc/(float)nImg)*100) +" %");
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
    else if (loss_type == "categorical_crossentropy"){
      float loss = categorical_crossentropy(estimated, y_);
      return loss;
    }
    println("ERROR: Loss_type missmatch");
    return -1; 
  }
  
  void genes2weights(float[] chromosome, int [] neu, NN_Model mod){
  float [][][] w = new float[neu.length-1] [] [];
  for(int n = 0; n < (neu.length-1); n++){
    w[n] = new float[neu[n+1]] [neu[n]];
  }
  
  int i = 0;
  int [] j = new int [neu.length-1];
  int [] k = new int [neu.length-1];
  
  for(int l = 0; l < (neu.length-1); l++){
    for (j[l] = 0; j[l] < neu[l+1]; j[l]++){
      for(k[l] = 0; k[l] < neu[l]; k[l]++){
        w [l] [j[l]][k[l]] = chromosome[i];
        i++;
      }
    }
  }
  //println(i);
  for(int m = 0; m < (neu.length-1); m++){
    mod.layers.get(m+1).setWeights(w[m]);
  }
  }
  
  void readWeights(int [] neu, NN_Model mod) {
    BufferedReader reader = createReader("Data\\WEIGHTS.txt");
    String line = null;
    try {
      while ((line = reader.readLine()) != null) {
        String [] w = split(line, TAB);
        float [] w_f = new float [w.length];
        for(int i = 0; i < w.length; i++){
          w_f[i] = Float.parseFloat(w[i]);
        }
        //printArray(w_f);
        genes2weights(float(w), neu, mod);
      }
      reader.close();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
  } 

}


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

float categorical_crossentropy(float[] z, float[]y_){
  float sum = 0; //<>//
  int n = z.length;
  for (int i = 0; i < n; i++){
    sum += log(z[i]) * y_[i];
  }
  return -sum;
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

float [] normalizacion(float [] num){
  float [] norm = new float[num.length];
  for(int i = 0; i < num.length; i++){
    norm [i] = num[i]/255;
  }
  return norm;
}
