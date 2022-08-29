package NeuralNetwork;

import processing.core.*;
import processing.core.PApplet;

public class Individual{
    // The parent Processing applet
	protected final PApplet parent;

    public int chromosome_length;
    public float[] chromosome;
    public float fitness;
    
    float chr_min, chr_max;
    
    public Individual(PApplet parent, int chro_length){
        this.parent = parent;
        chromosome_length = chro_length;
        chromosome = new float [chro_length];
        init(-1, 1);
    }
  
    void init(int min, int max){
        chr_min = min;
        chr_max = max;
        
        for(int i = 0; i < chromosome_length; i++){
            chromosome [i] = parent.random(min, max);
        }
    }
  
    public void addMutation(float mutation_rate){
        for(int i = 0; i < chromosome_length; i++){
            if (parent.random(1) < mutation_rate){
                chromosome [i] = parent.random(chr_min, chr_max);
                //parent.println("mutation gene: " + parent.str(i));
            }
        }
    }
  
    void printReport(){
        for(int i = 0; i < chromosome_length; i++){
            parent.print(parent.str(chromosome [i]) + " ");
        }
        parent.print("-> ");
        parent.println(fitness);
    }
}
