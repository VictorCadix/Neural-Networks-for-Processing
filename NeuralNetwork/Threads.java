package NeuralNetwork;

//import GeneticAlgorithm.*;
import processing.core.*;
import processing.core.PApplet;
import java.util.ArrayList; 

public class Threads extends Thread{
  // The parent Processing applet
  protected final PApplet parent;
  
  public int threadNum;
  public int nInd;
  public int last_im;
  public int batch_size;
  public int [] neu_;
  public float [][] x_;
  public float [][] y_;
  public ArrayList <Individual> ind;
  NN_Model mod;
  InputLayer in_;
  HiddenLayer [] lay_;
  OutputLayer out_;
  
  public Threads(PApplet parent, int num, int nInd, int last, int batch, int [] neu, ArrayList <Individual> ind, NN_Model model, float [][] x, float [][] y){
    this.parent = parent;
    this.threadNum = num;
    this.nInd = nInd;
    this.last_im = last;
    this.batch_size = batch;
    this.ind = ind;
    this.x_ = x;
    this.y_ = y;
    this.mod = new NN_Model(this.parent);
    this.neu_ = neu;
    this.lay_ = new HiddenLayer[model.layers.size()-2];
    
    for(int i = 0; i < model.layers.size(); i++){
      if( i == 0){
        in_ = new InputLayer(this.parent, neu[i]);
        mod.addLayer(in_);
      }
      else if(i ==  model.layers.size()-1 ){
        out_ = new OutputLayer(this.parent, neu[model.layers.size()-1], lay_[i-2], "softmax");
        mod.addLayer(out_);
      }
      else{
        if (i == 1){
          lay_[i-1] = new HiddenLayer(this.parent, neu[i], in_, "relu");
        }
        else{
          lay_[i-1] = new HiddenLayer(this.parent, neu[i], lay_[i-2], "relu");
        }
        mod.addLayer(lay_[i-1]);
      }
    }
    mod.setLoss("categorical_crossentropy");
  }
  
  public void run(){
    evaluar();
  }
  
  public void evaluar (){
    for (Individual indiv: ind){
      mod.genes2weights(indiv.chromosome, neu_, mod);
      float error = (float)0.0;
    
      //Procesa el batch
      for(int i = last_im; i < last_im + batch_size; i++) {
        in_.setNeurons(x_[i]);
        mod.forward_prop();
        error += mod.compute_loss(y_[i]);
      }
    
      error /= batch_size;
      indiv.fitness = 1/error;
    }
    //println("Hilo " + threadNum + " ha acabado");
  }
}
