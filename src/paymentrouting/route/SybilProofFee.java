package paymentrouting.route;

public class SybilProofFee {
    double rate = 0.0;
    SybilProofFee(double rate)
    {
        this.rate = rate;
    }
    public double charge(double size)
    {
        return size*this.rate;
    }
}
