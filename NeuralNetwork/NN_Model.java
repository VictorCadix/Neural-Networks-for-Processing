package NeuralNetwork;

//import GeneticAlgorithm.*;
import processing.core.*;
import processing.core.PApplet;
import java.util.ArrayList;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.BufferedReader;

public class NN_Model{
  // The parent Processing applet
  protected final PApplet parent;
  
  public ArrayList <Layer> layers;
  public String loss_type;
  public PrintWriter log_file, log_file2, log_file3, log_file4;
  
  public NN_Model(PApplet parent){
    this.parent = parent;
    layers = new ArrayList<Layer>();
    loss_type = "";
  }
  
  public void addLayer(Layer new_layer){
    layers.add(new_layer);
  }
  
  public void printParams(){
    parent.print("number of layers: ");
    parent.println(layers.size());
    
    int i = 0;
    for(Layer layer : layers){
      parent.print("layer " + parent.str(i) + " -> ");
      layer.printParams();
      i++;
    }
    parent.print("Total number of parameters: ");
    int nParams = 0;
    for(Layer layer : layers){
      nParams += layer.nParameters;
    }
    parent.println(nParams);
    
    parent.print("Loss function: ");
    parent.println(loss_type);
    parent.println("----------------------");
  }
  
   public void creatFiles(){
    String name = "LOSS";
    log_file = parent.createWriter("Data/" + name + ".txt");
    log_file.println("Generation,best_fitness");
    
    String name2 ="WEIGHTS";
    log_file2 = parent.createWriter("Data/" + name2 + ".txt");
  }
  
  public void creatFilesTest(){
    String name = "TEST";
    log_file3 = parent.createWriter("Data/" + name + ".txt");
    log_file3.println("Test MNIST:");
  }
  
  public void creatFilesValidation(){
    String name = "VALIDATION";
    log_file4 = parent.createWriter("Data/" + name + ".txt");
    log_file4.println("Validation MNIST:");
  }
  
  public void saveParamsLoss(int generation, int best, float fitness){
    log_file.print(generation);
    log_file.print(",");
    log_file.println(fitness);
    //log_file.println(pop.individuals[best].fitness);
  }
  
  public void ParamsWeights(int best, int chr_length, float [] chromosome){ 
    //int leng = pop.individuals[best].chromosome_length;
    for(int i= 0; i < chr_length; i++){
      log_file2.print(chromosome[i]);
    //log_file2.print(pop.individuals[best].chromosome[i]);
    //log_file2.print(weights_byte[i]);
      if(i < (chr_length-1)){
        log_file2.print("\t");
      }
    }
    exit1();
  }
  
  public void testFiles(int generacion, int seq, int num, float prob, int num_real){
    log_file3.println("Generacion: " + generacion + "\t" + "Secuencia: " + seq + "\t" + "Numero red: " + num + "\t" + "Probabilidad: " + prob + "\t" + "Numero Real: " + num_real);
  }
  
  public void testFilesNavigation(int generacion, int seq, float dist){
    log_file3.println("Generacion: " + generacion + "\t" + "Secuencia: " + seq + "\t" + "Distancia: " + dist);
  }
  
  public void sucess(int nImg, int suc){
     log_file3.println("El exito de la red ha sido de: " + (((float)suc/(float)nImg)*100) +" %");
     log_file3.println();
     log_file3.println();
  }
  
  public void exit1(){
    log_file.flush();
    log_file.close();
    parent.println("Archivo 1 cerrado");
  }
  
  public void exit2(){
    log_file2.flush();
    log_file2.close();
    parent.println("Archivo 2 cerrado");
  }
  
  public void exit3(){
    log_file3.flush();
    log_file3.close();
    parent.println("Archivo 3 cerrado");
  }
  
  public void exit4(){
    log_file4.flush();
    log_file4.close();
    parent.println("Archivo 4 cerrado");
  }
  
  public void setLoss(String loss_type){
    this.loss_type = loss_type;
  }
  
  public void forward_prop(){
    for (int i = 1; i < layers.size(); i++){
      Layer layer = layers.get(i);
      layer.compute_output();
      layer.activate();
    }
  }
  
  public float compute_loss(float[] y_){
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
    parent.println("ERROR: Loss_type missmatch");
    return -1; 
  }
  
  public void genes2weights(float[] chromosome, int [] neu, NN_Model mod){
    float [][][] w = new float[neu.length-1] [] [];
    for(int n = 0; n < (neu.length-1); n++){
      w[n] = new float[neu[n+1]] [neu[n]+1];
      //w[n] = new float[neu[n+1]] [neu[n]];
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
    for(int m = 0; m < (neu.length-1); m++){
      mod.layers.get(m+1).setWeights(w[m]);
    }
  }
  
 public void readWeights(int [] neu, NN_Model mod) {
    BufferedReader reader = parent.createReader("Data\\WEIGHTS.txt");
    String line = null;
    try {
      while ((line = reader.readLine()) != null) {
        String [] w = parent.split(line, parent.TAB);
        float [] w_f = new float [w.length];
        for(int i = 0; i < w.length; i++){
          w_f[i] = Float.parseFloat(w[i]);
        }
        //printArray(w_f);
        mod.genes2weights(w_f, neu, mod);
      }
      reader.close();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
  } 
  //La capa de salida debe de calcular el costo de cada simulación. 
  //Para ello, se hace el cuadrado del valor que se obtiene en cada 
  //neurona de salida restando el valor que se desea obtener en cada 
  //una de estas neuronas

  public float mae(float[] z, float[]y_){
    float mae = 0;
    int n = z.length;
    for (int i = 0; i < n; i++){
      mae += parent.abs(z[i]-y_[i]);
    }
    return mae/n;
  }

  public float mse(float[] z, float[]y_){
    float mse = 0;
    int n = z.length;
    for (int i = 0; i < n; i++){
      mse += parent.pow(z[i]-y_[i], 2);
    }
    return mse/n;
}

  public float categorical_crossentropy(float[] z, float[]y_){
    float sum = 0;
    int n = z.length;
    for (int i = 0; i < n; i++){
      sum += parent.log((float)(z[i] + 1e-10)) * y_[i];
    }
    return -sum;
}

  public float func_costo(float z[],int want){
    float costo = 0;
    int n = z.length;
    for (int i = 0; i < n; i++){
      if (n == want)
      costo += parent.pow(z[i]-1, 2);
      else
      costo += parent.pow(z[i]-0, 2);  
    }
    return costo;
}

  public static byte[] floatToByteArray(float value) {
    int intBits =  Float.floatToIntBits(value);
    return new byte[] {
      (byte) (intBits >> 24), (byte) (intBits >> 16), (byte) (intBits >> 8), (byte) (intBits) };
  }

  public float [] normalizacion(float [] num){
    float [] norm = new float[num.length];
    for(int i = 0; i < num.length; i++){
      norm [i] = num[i]/255;
    }
    return norm;
  }
}
