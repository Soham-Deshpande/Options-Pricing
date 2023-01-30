
class BinomialAsssetPricingModel{

  
  public void getPrice(int S1P, int S1L, int S0, double r, double delta, double X0){

    System.out.println("S1P: "+S1P);
    System.out.println("S1L: "+S1L);
    System.out.println("S0: "+S0);
    System.out.println("Interest rate: "+r);
    System.out.println("Delta: "+delta);
    System.out.println("Initial Wealth: "+X0);
    double success_price = 0;
    double loss_price = 0;
    success_price = 0.5*(S1P) + (1+r)*(X0-(delta*S0));
    loss_price = 0.5*(S1L) + (1+r)*(X0-(delta*S0));
    System.out.println("Success: "+ success_price);
    System.out.println("Loss: "+ loss_price); 
  }
  
  public static void main(String[] args){
  
    BinomialAsssetPricingModel Binomial = new BinomialAsssetPricingModel();
    Binomial.getPrice(8,2,4,0.25,0.5,1.2);

  }

}
