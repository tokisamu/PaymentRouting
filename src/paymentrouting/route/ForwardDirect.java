package paymentrouting.route;

import java.util.*;

import gtna.graph.Graph;
import gtna.graph.Node;

import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * only split if necessary
 * if so: split as few times as possible by using neighbors with highest balances
 * @author mephisto
 *
 */
public class ForwardDirect extends PathSelection {
    ClosestNeighbor cn;

    public ForwardDirect(DistanceFunction df) {
        super("SPLIT_IFNECESSARY", df);
        this.cn = new ClosestNeighbor(df); //ClosestNeighbor is used when not splitting
    }

    @Override
    public void initRoutingInfo(Graph g, Random rand) {
        this.dist.initRouteInfo(g, rand);

    }


    @Override
    public double[] getNextsVals(Graph g, int cur, int dst, int pre, boolean[] excluded,
                                 RoutePayment rp, double curVal,
                                 Random rand, int reality) {
        //check if not splitting work (using ClosestNeighbor), otherwise go to splitting
        double[] noSplit = this.cn.getNextsVals(g, cur, dst, pre, excluded, rp, curVal, rand, reality);
        if (noSplit != null) {
            return noSplit;
        }
        Node[] nodes = g.getNodes();
        int[] out = nodes[cur].getOutgoingEdges();
        //sum all funds that can be forwarded
        double sum = 0;
        double totalSum = 0.0;
        HashMap<Double, Vector<Integer>> pots = new HashMap<Double, Vector<Integer>>();
        for (int k = 0; k < out.length; k++) {
            if (out[k] == pre || excluded[out[k]]) continue;
            if (this.dist.isCloser(out[k], cur, dst, reality)) {
                double pot = rp.computePotential(cur, out[k]);
                Vector<Integer> vec = pots.get(pot);
                if (vec == null) {
                    vec = new Vector<Integer>();
                    pots.put(pot, vec);
                }
                vec.add(k);
                totalSum+= rp.computePotential(out[k],cur);
                sum = sum + pot;
            }
        }

        //sort nodes by potential (available funds)
        double[] partVal = new double[out.length];
        for(int i=0;i<partVal.length;i++)
            partVal[i] = 0.0;
        Iterator<Double> it = pots.keySet().iterator();
        double[] vals = new double[pots.size()];
        int c = 0;
        while (it.hasNext()) {
            vals[c] = it.next();
            c++;
        }
        Arrays.sort(vals);
        double all = 0;
        if(vals.length>0)
            if(vals[0]>curVal)
            {
                Vector<Integer> vec = pots.get(vals[0]);
                int node = vec.remove(rand.nextInt(vec.size()));
                double valNode = curVal;
                partVal[node] = valNode;
            }
        return partVal;

    }

}
