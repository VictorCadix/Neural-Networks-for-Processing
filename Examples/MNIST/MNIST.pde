import NeuralNetwork.*;

NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;
int [] neu;
Population population;
int nParameters = 0;
int nIndiv = 100;
int nCrossPoints = 20;
float mutation_rate = 0.005;
int elite_indivs = 25;

float min_error;
int generation = 0;
//Raw dataset
byte [] imag,num;

//Process dataset
float [][] img_int;
int [] num_int;
float [][] x_train;
float [][] y_train;
float [][] x_val;
float [][] y_val;
int nImg;

//Training
int batch_size = 200;
int last_img = 0;
int maxGenerations = 1000;
int epoch = 0;

//Validation
float validation_sample = 0.1;
int nSmples_train = 0;
int nSmples_val = 0;
boolean do_validation = false;
int sum = 0;
int nums = 0;
float prob = 0.0;

void setup(){
  size(800,400);
  
  model = new NN_Model(this);
  neu = new int [4];
  neu[0] = 784; neu[1] = 392; neu[2] = 392; neu[3]= 10;
  
  in = new InputLayer(this, neu[0]);
  lay1 = new HiddenLayer(this, neu[1], in, "tanh");
  lay2 = new HiddenLayer(this, neu[2], lay1, "tanh");
  out = new OutputLayer(this, neu[3], lay2, "tanh");
  
  for(int i = 0; i < (neu.length-1); i++){
    nParameters += neu[i]*neu[i+1];
  }
  //println(nParameters);
  population = new Population(this, nIndiv, nParameters);
  population.crossover_type = "multiple_random";
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  model.setLoss("categorical_crossentropy");
  
  model.printParams(); 
  model.creatFiles();
  model.creatFilesTest();
  model.creatFilesValidation();
  
  imag = loadBytes("train-images.idx3-ubyte");
  num = loadBytes("train-labels.idx1-ubyte");
  
  num_int = new int[num.length - 8];
  for (int i = 8 ; i < num.length; i++) {
    num_int [i-8] = (num [i] & 0xFF);
  }
  print("num_int.length: ");
  println(num_int.length);
  
  nImg = (imag.length - 16) / (28*28); 
  print("nImg: ");
  println(nImg);
   
  img_int = new float[nImg][28*28];
  for (int i = 0; i < nImg; i++) {
    for(int j = 0; j < 784; j++){
      img_int [i][j] = (imag [(i*784)+j+16] & 0xFF);
    }
  }
  
  nSmples_train = int(nImg * (1 - validation_sample));
  nSmples_val = int(nImg * validation_sample);

  //Normalize the data
  x_train = new float[nSmples_train][28*28];
  for (int i = 0; i < nSmples_train; i++){
    x_train[i] = model.normalizacion(img_int [i]);
  }
  x_val = new float[nSmples_val][28*28];
  for (int i = 0; i < nSmples_val; i++){
    x_val[i] = model.normalizacion(img_int [nSmples_train + i]);
  }
  
  //Create output vector
  y_train = new float[nSmples_train][10];
  for (int i = 0 ; i < nSmples_train; i++) {
    int digit = num_int[i];
    y_train[i][digit] = 1;
  }
  y_val = new float[nSmples_val][10];
  for (int i = 0 ; i < nSmples_val; i++) {
    int digit = num_int[nSmples_train + i];
    y_train[i][digit] = 1;
  }
  
  
  //print(num_int[0]);
  //println(" -> to y_train");
  //println(y_train[0]);

  /*
  print("img_int[0]: ");
  for (int i = 0; i < 784; i++){
    if (i % 28 == 0){
      println();
    }
    print(img_int[0][i]);
    print(",");
  }
  println();
  */  
  
  
}

void draw(){
  /*
  PImage disp_img = createImage(28, 28, RGB);
  disp_img.loadPixels();
  for (int i = 0; i < disp_img.pixels.length; i++) {
    float pixel_val = img_int[1][i];
    disp_img.pixels[i] = color(pixel_val,pixel_val,pixel_val);
  }
  disp_img.updatePixels();
  image(disp_img, 0, 0);
  */
  
  generation++;
  println("Generation: " + str(generation));
  
  //Comprueba que hay un batch completo
  //si no, comienza de nuevo
  if (last_img + batch_size > (nSmples_train)){
    last_img = 0;
    epoch++;
    println("Epoch " + str(epoch));
    do_validation = true;
  }
  //int flat = 0;
  //Evaluate
  for (Individual indiv: population.individuals){
    model.genes2weights(indiv.chromosome, neu, model);
    float error = 0.0;
    
    //Procesa el batch
    for(int i = last_img; i < last_img + batch_size; i++) {
      in.setNeurons(x_train[i]);
      model.forward_prop();
      error += model.compute_loss(y_train[i]);
    }
    //println ("NEW");
    //println("ERROR: " + error);
    error /= batch_size;
    indiv.fitness = 1/error;
    //flat++;
    //println("INDIVIDUO: " +  flat +"FIT: " + indiv.fitness);
  }
  
  population.calculate_selection_probability();
  
  //metrics
  int best = population.getBetsIndiv();
  model.genes2weights(population.individuals[best].chromosome, neu, model);
  sum = 0;
  float mse_ = 0;
  for(int i = last_img; i < last_img + batch_size; i++) {
    in.setNeurons(x_train[i]);
    model.forward_prop();
    //mse
    mse_ += model.mse(model.layers.get(model.layers.size()-1).neurons, y_train[i]);
    //accuracy
    nums = num_correct(model.layers.get(model.layers.size()-1).nNeurons,model.layers.get(model.layers.size()-1).neurons);
    if (nums == num_int[i]){
      sum++;
    }
  }
  float accuracy = float(sum) / batch_size;
  print(population.individuals[best].fitness);
  println(" -> " + str(accuracy*100) + "% " + "mse: " + str(mse_ / batch_size));
  
  last_img = last_img + batch_size;

  model.saveParamsLoss(generation, best, population.individuals[best].fitness);
  
  if (do_validation){
    do_validation = false;
    print("Validation");
    model.genes2weights(population.individuals[best].chromosome, neu, model);
    float error = 0;
    for(int i = 0; i < nSmples_val; i++){
      in.setNeurons(x_val[i]);
      model.forward_prop();
      nums = num_correct(model.layers.get(model.layers.size()-1).nNeurons,model.layers.get(model.layers.size()-1).neurons);
      prob = prob_numcorrect(model.layers.get(model.layers.size()-1).nNeurons,model.layers.get(model.layers.size()-1).neurons);
      model.testFiles(generation, i+1,nums,prob,num_int[i]);
      if (nums == num_int[i]){
        sum++;
      }
      error += model.compute_loss(y_train[i]);
    }
    error /= nSmples_val;
    print(" : " + str(error));
    accuracy = float(sum) / nSmples_val * 100;
    println(" -> " + str(accuracy) + "%");
    model.sucess(nImg, sum);
  }
  
Individual child [] = new Individual [nIndiv];
    
    //Elitism
    if(elite_indivs != 0){
      int [] best_indivs = new int[elite_indivs];
      best_indivs[0] = best;
      float last_best_fitness = population.individuals[best].fitness;
      float best_fitness;
      for (int ei = 0; ei < elite_indivs; ei++){
        best_fitness = 0;
        for (int i = 0; i < nIndiv; i++){
          if (population.individuals[i].fitness > best_fitness){
            if (population.individuals[i].fitness < last_best_fitness){
              best_fitness = population.individuals[i].fitness;
              best_indivs[ei] = i;
            }
          }
        }
        last_best_fitness = best_fitness;
      }    
    
      for (int i = 0; i < elite_indivs; i++){
        //print("best_indivs:" + best_indivs[i]);
        //println("\tfitness:" + population.individuals[best_indivs[i]].fitness);
        child[i] = population.individuals[best_indivs[i]]; 
      }
    }
    
    for (int i = elite_indivs; i < nIndiv; i++){
      int p1 = population.get_parent();
      int p2 = population.get_parent();
    
      //crossover
      child[i] = population.crossover(p1, p2, nCrossPoints);
      //mutation
      child[i].addMutation(mutation_rate);
    }
  
    // Renew population
    for (int i = 0; i < nIndiv; i++){
      population.individuals[i] = child[i];  
    }
  
  if (generation == maxGenerations){
    model.ParamsWeights(best, population.individuals[best].chromosome_length, population.individuals[best].chromosome);
    model.exit2();
    model.exit3();
    model.exit4();
    super.exit();//let processing carry with it's regular exit routine
  }
}

int num_correct(int nNeurons, float neurons[]){
  int index = 0;
  for (int i = 1; i < nNeurons; i++){
    if (neurons[i] > neurons[index]){
      index = i;
    }
  }
  return index;
}

float prob_numcorrect(int nNeurons, float neurons[]){
  int index = 0;
  for (int i = 1; i < nNeurons; i++){
    if (neurons[i] > neurons[index]){
      index = i;
    }
  }
  return neurons[index];
}
