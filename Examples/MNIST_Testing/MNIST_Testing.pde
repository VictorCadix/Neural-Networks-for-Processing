NN_Model model;
InputLayer in;
HiddenLayer lay1,lay2;
OutputLayer out;
int neuin = 784, neulay1 = 18, neulay2 = 18, neuout = 10;
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
    norm_img_int[i] = normalizacion(img_int[i]);
  }
  
  model = new NN_Model();
  in = new InputLayer(neuin);
  lay1 = new HiddenLayer(neulay1, in, "relu");
  lay2 = new HiddenLayer(neulay2, lay1, "relu");
  out = new OutputLayer(neuout, lay2, "sigmoid");
  
  model.addLayer(in);
  model.addLayer(lay1);
  model.addLayer(lay2);
  model.addLayer(out);
  
  model.printParams();
  model.readWeights();
  model.creatFilesTest();
  

  for(int i = 0; i < nImg; i++){
    in.setNeurons(norm_img_int[i]);
    model.forward_prop();
    nums = out.numMNIST();
    prob = out.prob_numMNIST();
    model.testFiles(i+1,nums,prob,num_int[i], out);
    
    if (nums == num_int[i]){
      sum++;
    }
  }
  model.sucess(nImg, sum);
  model.exit2();
  super.exit();
}

void draw(){
  
}
