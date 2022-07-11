
class Population{
  int nIndividues;
  float mutation_rate;
  
  Individual[] individuals;
  float[] probability;
  
  String crossover_type = "";
  
  Population(int number_individues, int nParameters){
    nIndividues = number_individues;
    individuals = new Individual[nIndividues];
    for (int i = 0; i < nIndividues; i++){
      individuals[i] = new Individual(nParameters);
    }
    probability = new float[number_individues];
    
    //defaults
    crossover_type = "one_point";
  }
  
  void calculate_selection_probability() {
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
  
  int get_parent() {
    float target = random(100);
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
    
    println("ERROR: get parent index");
    println("target: " + str(target));
    println("accum_prob: " + str(accum_prob));
    return -1; //Error
  }
  
  Individual crossover(int parent1, int parent2, int nPoints){
    Individual child = new Individual(nParameters);
    if (crossover_type == "multiple_random"){
      child = multiple_random_crossover(parent1, parent2, nPoints);
    }
    else if (crossover_type == "one_point"){
      child = one_point_crossover(parent1, parent2);
    }
    else{
      println("crossover_type ERROR");
    }
    return child;
  }
  
  Individual multiple_random_crossover(int parent1, int parent2, int nPoints){
    int nParameters = individuals[parent1].chromosome_length;
    Individual child = new Individual(nParameters);

    int chunk_size = nParameters / nPoints;
    int last_point = 0;
    
    for (int point = 0; point < nPoints; point++){
      int parent = (point % 2 == 0)? parent1 : parent2;
      
      int crossover_point = round(random(last_point, (point+1) * chunk_size));
      //crossover_point = (crossover_point >= nParameters-1)? nParameters : crossover_point;
      
      for (int i = last_point; i < crossover_point; i++){
        child.chromosome[i] = individuals[parent].chromosome[i];
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
    Individual child = new Individual(nParameters);
    
    int crossover_point = round(random(0, nParameters));
    
    for (int i = 0; i < crossover_point; i++){
      child.chromosome[i] = individuals[parent1].chromosome[i];
    }
    for (int i = crossover_point; i < nParameters; i++){
      child.chromosome[i] = individuals[parent2].chromosome[i];
    }
    return child;
  }
  
  int getBetsIndiv(){
    int index = 0;
    
    for (int i = 1; i < nIndividues; i++){
      if (individuals[i].fitness > individuals[index].fitness){
        index = i;
      }
    }
    return index;
  }
  
  void printReport(){
    for (int i = 0; i < nIndividues; i++){
      print("[" + str(i) + "] ");
      individuals[i].printReport();
    }
  }
    
}

class Individual{
  int chromosome_length;
  float[] chromosome;
  float fitness;
  float loss;
  
  float chr_min, chr_max;
  
  Individual(int chro_length){
    chromosome_length = chro_length;
    chromosome = new float [chro_length];
    init(-1, 1);
  }
  
  void init(int min, int max){
    chr_min = min;
    chr_max = max;
    
    for(int i = 0; i < chromosome_length; i++){
      chromosome [i] = random(min, max);
    }
  }
  
  void addMutation(float mutation_rate){
    for(int i = 0; i < chromosome_length; i++){
      if (random(1) < mutation_rate){
        chromosome [i] = random(chr_min, chr_max);
      }
    }
  }
  
  void printReport(){
    for(int i = 0; i < chromosome_length; i++){
      print(str(chromosome [i]) + " ");
    }
    print("-> ");
    println(fitness);
  }
}
