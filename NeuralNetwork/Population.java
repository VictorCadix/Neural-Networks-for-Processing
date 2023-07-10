package NeuralNetwork;

import processing.core.*;
import processing.core.PApplet;

public class Population{
    // The parent Processing applet
    protected final PApplet parent;

    public int nIndividues;
    float mutation_rate;
    
    public Individual[] individuals;
    public float[] probability;
    
    public String crossover_type = "";
  
    public Population(PApplet parent, int number_individues, int nParameters){
        this.parent = parent;
        nIndividues = number_individues;
        individuals = new Individual[nIndividues];
        for (int i = 0; i < nIndividues; i++){
            individuals[i] = new Individual(parent, nParameters);
        }
        probability = new float[number_individues];
        
        //defaults
        crossover_type = "one_point";
    }
  
    public void calculate_selection_probability() {
        float sum = 0;
        for (Individual indiv: individuals) {
            sum += indiv.fitness;
        }
        int i = 0;
        for (Individual indiv: individuals) {
            probability[i] = indiv.fitness / sum * 100;
            i++;
        }
    }
  
    public int get_parent() {
        float target = parent.random(100);
        float accum_prob = 0;
        for (int i = 0; i < nIndividues; i++) {
            accum_prob += probability[i];
            if (accum_prob >= target) {
                return i;
            }
        }
        if (target <= 100){
            return nIndividues -1;
        }
    
        parent.println("ERROR: get parent index");
        parent.println("target: " + parent.str(target));
        parent.println("accum_prob: " + parent.str(accum_prob));
        return -1; //Error
    }
  
    public Individual crossover(int parent1, int parent2, int nPoints){
        Individual child = new Individual(this.parent, individuals[parent1].chromosome_length);
        if (crossover_type == "multiple_random"){
            child = multiple_random_crossover(parent1, parent2, nPoints);
        }
        else if (crossover_type == "one_point"){
            child = one_point_crossover(parent1, parent2);
        }
        else{
            parent.println("crossover_type ERROR");
        }
        return child;
    }
  
    Individual multiple_random_crossover(int parent1, int parent2, int nPoints){
        int nParameters = individuals[parent1].chromosome_length;
        Individual child = new Individual(this.parent, nParameters);

        int chunk_size = nParameters / nPoints;
        int last_point = 0;
        
        for (int point = 0; point < nPoints; point++){
            int parent_n = (point % 2 == 0)? parent1 : parent2;
            
            int crossover_point = parent.round(parent.random(last_point, (point+1) * chunk_size));
            //crossover_point = (crossover_point >= nParameters-1)? nParameters : crossover_point;
        
            for (int i = last_point; i < crossover_point; i++){
                child.chromosome[i] = individuals[parent_n].chromosome[i];
            }
            last_point = crossover_point;
            
            if (crossover_point == nParameters){
                break;
            }
        }
        for (int i = last_point; i < nParameters; i++){
        child.chromosome[i] = individuals[parent2].chromosome[i];
        }
        return child;
    }
  
    Individual one_point_crossover(int parent1, int parent2){
        int nParameters = individuals[parent1].chromosome_length;
        Individual child = new Individual(parent, nParameters);
        
        int crossover_point = parent.round(parent.random(0, nParameters));
        
        for (int i = 0; i < crossover_point; i++){
            child.chromosome[i] = individuals[parent1].chromosome[i];
        }
        for (int i = crossover_point; i < nParameters; i++){
            child.chromosome[i] = individuals[parent2].chromosome[i];
        }
        return child;
    }
  
    public int getBetsIndiv(){
        int index = 0;
        
        for (int i = 1; i < nIndividues; i++){
            if (individuals[i].fitness > individuals[index].fitness){
                index = i;
            }
        }
        return index;
    }
  
    public void printReport(){
        for (int i = 0; i < nIndividues; i++){
            parent.print("[" + parent.str(i) + "] ");
            individuals[i].printReport();
        }
    }
    
}
