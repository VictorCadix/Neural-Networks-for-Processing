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
    log_file.println("Generation,best_fitness");
    
    String name2 ="WEIGHTS";
    log_file2 = createWriter("Data/" + name2 + ".txt");
    //log_file2.println("Weights best individual:");
  }
  
  void creatFilesTest(){
    String name = "TEST";
    log_file3 = createWriter("Data/" + name + ".txt");
    log_file3.println("Test MNIST:");
  }
  
  void creatFilesValidation(){
    String name = "VALIDATION";
    log_file3 = createWriter("Data/" + name + ".txt");
    log_file3.println("Validation MNIST:");
  }
  
  void saveParamsLoss(int generation, int best, Population pop){
    log_file.print(generation);
    log_file.print(",");
    log_file.println(pop.individuals[best].fitness);
  }
  
  void ParamsWeights(int best, Population pop){ 
    //log_file2.print("{");
    for(int i= 0; i < pop.individuals[best].chromosome_length; i++){
    log_file2.print(pop.individuals[best].chromosome[i]);
    //log_file2.print(weights_byte[i]);
      if(i < (pop.individuals[best].chromosome_length-1)){
        log_file2.print("\t");
      }
    }
    //log_file2.print(" }FINISH");
    exit1();
  }
  
  void testFiles(int generacion, int seq, int num, float prob, int num_real){
    log_file3.println("Generacion: " + generacion + "\t" + "Secuencia: " + seq + "\t" + "Numero red: " + num + "\t" + "Probabilidad: " + prob + "\t" + "Numero Real: " + num_real);
  }
  
  void sucess(int nImg, int suc){
     log_file3.println("El exito de la red ha sido de: " + (((float)suc/(float)nImg)*100) +" %");
     log_file3.println();
     log_file3.println();
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
      /*for(int j = 0; j < layer.nNeurons; j++){
      if (Float.isNaN(layer.neurons[j])){
         println("Capa: " + i);
         println("Neurona: " + j);
        println(layer.neurons[j]);
      }
      }//*/
    }
  }
  
  float compute_loss(float[] y_){
    int output_layer = layers.size() - 1;
    float[] estimated = layers.get(output_layer).neurons;
    //printArray(estimated);
    
    /*for(int i=0; i<estimated.length;i++){
      if (Float.isNaN(estimated[i])){
         print("Estimated: ");
        printArray(estimated);
      }
    }*/

    //printArray(estimated);
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
}


//La capa de salida debe de calcular el costo de cada simulaci??n. 
//Para ello, se hace el cuadrado del valor que se obtiene en cada 
//neurona de salida restando el valor que se desea obtener en cada 
//una de estas neuronas

float mae(float[] z, float[]y_){
  float mae = 0;
  int n = z.length;
  for (int i = 0; i < n; i++){
    mae += abs(z[i]-y_[i]);
    //if (Float.isNaN(mae)){
    //println("Z: " + z[i] + "Y: " + y_[i]);
    //println("MAE: " + mae);}
  //}
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
    sum += log(z[i] + 1e-30) * y_[i];
    //println("SUM: " + sum);
    //println("Z: " + z[i] + "Y: " + y_[i]);
  }
  //println("SUM: " + sum);
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
