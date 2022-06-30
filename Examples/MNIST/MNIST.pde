NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;
int neuin = 784, neulay1 = 18, neulay2 = 18, neuout = 10;
Population population;
int nParameters;
int nIndiv = 1000;
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
int nImg;

//Training
int batch_size = 100;
int last_img = 0;

void setup(){
  //size(800,400);
  imag = loadBytes("train-images.idx3-ubyte");
  num = loadBytes("train-labels.idx1-ubyte");
  
  num_int = new int[num.length - 8];
  print("num.length: ");
  println(num.length);
  
  for (int i = 8 ; i < num.length; i++) {
    num_int [i-8] = (num [i] & 0xFF);
  }
  print("num_int.length: ");
  println(num_int.length);
  println(num_int[0]);
  
  nImg = (imag.length - 16) / (28*28); 
  print("nImg: ");
  println(nImg);
  img_int = new float[nImg][28*28];
  for (int i = 0; i < nImg; i++) {
    for(int j = 0; j < 784; j++){
      img_int [i][j] = (imag [(i*784)+j+16] & 0xFF);
    }
  }
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
  out = new OutputLayer(neuout, lay2, "sigmoid");
  
  nParameters = neuin*neulay1 + neulay1*neulay2 + neulay2*neuout;
  population = new Population(nIndiv, nParameters);
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  
  model.printParams(); 
  model.creatFiles();
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
  print("Generation: ");
  println(generation);
  
  //Evaluate
  for (Individual indiv: population.individuals){
    model.genes2weights(indiv.chromosome,neulay1,neulay2,neuin,neuout,lay1,lay2,out);
    float error = 0.0;
    
    //Comprueba que hay un batch completo
    //si no, comienza de nuevo
    if (last_img + batch_size > nImg){
      last_img = 0;
    }
    //Procesa el batch
    for(int i = last_img; i < last_img + batch_size; i++) {
      in.setNeurons(normalizacion(img_int [i]));
      model.forward_prop();
      error += func_costo(out.neurons,num_int[i]);
    }
    error /= batch_size;
    //error += 0.001;
    indiv.fitness = 1/error;
    last_img = last_img + batch_size;
  }
  
  population.calculate_selection_probability();
  
  int best = population.getBetsIndiv();
  println(population.individuals[best].fitness);
  //System.out.println(String.format("%.5f", population.individuals[best].fitness));
  model.saveParamsLoss(generation, best, population);
  
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
  
  if (generation == 20){
    model.ParamsWeights(best, population);
    super.exit();//let processing carry with it's regular exit routine
  }
} //<>//
