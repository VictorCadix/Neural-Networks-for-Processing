import NeuralNetwork.*;

NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;
int [] neu;
int sum = 0;
int nums = 0;
float prob = 0.0;
//Raw dataset
byte [] imag,num;

//Process dataset
float [][] img_int;
float [][] norm_img_int;
int [] num_int;
int nImg;

void setup(){
  //size(800,400);
  model = new NN_Model(this);
  neu = new int [4];
  neu[0] = 784; neu[1] = 392; neu[2] = 392; neu[3]= 10;
  
  in = new InputLayer(this, neu[0]);
  lay1 = new HiddenLayer(this, neu[1], in, "tanh");
  lay2 = new HiddenLayer(this, neu[2], lay1, "tanh");
  out = new OutputLayer(this, neu[3], lay2, "tanh");
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  
  model.printParams();
  model.readWeights(neu, model);
  model.creatFilesTest();
  imag = loadBytes("t10k-images.idx3-ubyte");
  num = loadBytes("t10k-labels.idx1-ubyte");
  
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
  norm_img_int = new float[nImg][28*28];
  for (int i = 0; i < nImg; i++) {
    for(int j = 0; j < 784; j++){
      img_int [i][j] = (imag [(i*784)+j+16] & 0xFF);
    }
  }
  for (int i = 0; i < nImg; i++) {
    norm_img_int[i] = model.normalizacion(img_int[i]);
  }
  

  for(int i = 0; i < nImg; i++){
    in.setNeurons(norm_img_int[i]);
    model.forward_prop();
    nums = num_correct(model.layers.get(model.layers.size()-1).nNeurons,model.layers.get(model.layers.size()-1).neurons);
    prob = prob_numcorrect(model.layers.get(model.layers.size()-1).nNeurons,model.layers.get(model.layers.size()-1).neurons);
    model.testFiles(i+1,0,nums,prob,num_int[i]);
    
    if (nums == num_int[i]){
      sum++;
    }
  }
  model.sucess(nImg, sum);
  model.exit3();
  super.exit();
}

void draw(){
  
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
