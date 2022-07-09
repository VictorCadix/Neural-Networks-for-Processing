NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;
int neuin = 784, neulay1 = 18, neulay2 = 18, neuout = 10;
Population population;
int nParameters;
int nIndiv = 100;
int nCrossPoints = 1000;
float mutation_rate = 0.0001;
int elitism = 0;

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
int batch_size = 100;
int last_img = 0;
int maxGenerations = 100;
int epoch = 0;

//Validation
float validation_sample = 0.1;
int nSmples_train = 0;
int nSmples_val = 0;
boolean do_validation = true;
int sum = 0;
int nums = 0;
float prob = 0.0;

void setup(){
  size(800,400);
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
    x_train[i] = normalizacion(img_int [i]);
  }
  x_val = new float[nSmples_val][28*28];
  for (int i = 0; i < nSmples_val; i++){
    x_val[i] = normalizacion(img_int [nSmples_train + i]);
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
  
  model = new NN_Model();
  in = new InputLayer(neuin);
  lay1 = new HiddenLayer(neulay1, in, "relu");
  lay2 = new HiddenLayer(neulay2, lay1, "relu");
  out = new OutputLayer(neuout, lay2, "softmax");
  
  nParameters = neuin*neulay1 + neulay1*neulay2 + neulay2*neuout;
  population = new Population(nIndiv, nParameters);
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  model.setLoss("categorical_crossentropy");
  
  model.printParams(); 
  model.creatFiles();
  model.creatFilesValidation();
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
  
  //Evaluate
  for (Individual indiv: population.individuals){
    model.genes2weights(indiv.chromosome,neulay1,neulay2,neuin,neuout,lay1,lay2,out);
    float error = 0.0;
    
    //Procesa el batch
    for(int i = last_img; i < last_img + batch_size; i++) {
      in.setNeurons(x_train[i]);
      model.forward_prop();
      error += model.compute_loss(y_train[i]);
      //println ("ERROR:" + error);
    }
    //println ("NEW");
    error /= batch_size;
    indiv.loss = error;
    indiv.fitness = 1/error;
  }
  
  population.calculate_selection_probability();
  
  int best = population.getBetsIndiv();
  
  //accuracy
  model.genes2weights(population.individuals[best].chromosome, neulay1, neulay2, neuin, neuout,lay1,lay2,out);
  sum = 0;
  for(int i = last_img; i < last_img + batch_size; i++) {
    in.setNeurons(x_train[i]);
    model.forward_prop();
    nums = out.numMNIST();
    if (nums == num_int[i]){
      sum++;
    }
  }
  float accuracy = float(sum) / batch_size * 100;
  println(best);
  println(population.individuals[best].chromosome[8]);
  print(population.individuals[best].loss);
  println(" -> " + str(accuracy) + "%");
  
  last_img = last_img + batch_size;

  model.saveParamsLoss(generation, best, population);
  
  if (do_validation){
    do_validation = false;
    print("Validation");
    model.genes2weights(population.individuals[best].chromosome, neulay1, neulay2, neuin, neuout,lay1,lay2,out);
    float error = 0;
    for(int i = 0; i < nSmples_val; i++){
      in.setNeurons(x_val[i]);
      model.forward_prop();
      nums = out.numMNIST();
      prob = out.prob_numMNIST();
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
  for (int i = 0; i < nIndiv; i++){
    int p1 = population.get_parent();
    int p2 = population.get_parent();
    
    //println(str(p1) + ":" + str(p2));
    
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
    model.ParamsWeights(best, population);
    model.exit2();
    super.exit();//let processing carry with it's regular exit routine
  }
}
