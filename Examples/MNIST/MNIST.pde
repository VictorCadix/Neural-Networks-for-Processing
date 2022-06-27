NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;
int neuin = 36,neulay1 = 18,neulay2 = 18,neuout = 10;
Population population;
int nParameters;
int nIndiv = 10000;
int nCrossPoints = 1000;
float mutation_rate = 0.0001;
int elitism = 0;

float min_error;
PrintWriter log_file;
int generation = 0;

byte [] imag,num;

void setup(){
  size(800,400);
  imag = loadBytes("train-images-idx3-ubyte");
  num = loadBytes("train-labels-idx1-ubyte");
  int [] num_int = new int[num.length - 8];
  print("num.length: ");
  print(num.length);
  
  for (int i = 8 ; i < num.length; i++) {
    num_int [i-8] = (num [i] & 0xFF);
  }
  
  model = new NN_Model();
  //printArray(imag);
  //neuin = imag[1].size();
  in = new InputLayer(neuin);
  lay1 = new HiddenLayer(neulay1, in, "relu");
  lay2 = new HiddenLayer(neulay2, lay1, "relu");
  out = new OutputLayer(neuout, lay2, "sigmoid");
  
  nParameters = neuin*neulay1 + neulay1*neulay2 + neulay2*neuout;
  population = new Population(nIndiv, nParameters);
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  
  model.printParams();
  
  String name = "log_" + str(nIndiv) + "i_" + str(mutation_rate) + "m_" + str(nCrossPoints) + "cp_"  + str(elitism) + "E_#";
  log_file = createWriter("Data/" + name + ".txt");
  log_file.println("generation,best_fitness");
  
}

void draw(){
  generation++;
  print("Generation: ");
  println(generation);
  
  model.forward_prop();
  float[] y_ = {0.1, 0.9};
  //float loss = model.compute_loss(y_);
  //print("loss: ");
 // print(loss);
  population.calculate_selection_probability();
  
  Individual child [] = new Individual [nIndiv];
  for (int i = 0; i < nIndiv; i++){
    int p1 = population.get_parent();
    int p2 = population.get_parent();
    
    //println(str(p1) + ":" + str(p2));
    
    //crossover
    child[i] = population.crossover(p1, p2, nCrossPoints);
    
    //mutation
    child[i].addMutation(mutation_rate);
    
  }
  int best = population.getBetsIndiv();
  //PImage model = genes2image(population.individuals[best].chromosome);
  println(population.individuals[best].fitness);
  //image(target_image, 0, 0, 400, 400);
  //image(model, 400, 0, 400, 400);
  
  log_file.print(generation);
  log_file.print(",");
  log_file.println(population.individuals[best].fitness);
  
  
  // Renew population
  for (int i = 0; i < nIndiv; i++){
    population.individuals[i] = child[i];    
  }
  
}
