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
public class SplitAndAdjustSize extends PathSelection {
    ClosestNeighbor cn;

    public SplitAndAdjustSize(DistanceFunction df) {
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
        double maxCap = 0;
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
                maxCap = max(maxCap,pot);
                sum = sum + pot;
            }
        }
        totalSum += sum;
        double ratio = (sum - curVal) / (totalSum);
        if(ratio>0.25)
        {
            double[] partVal = new double[out.length];
            for (int i = 0; i < partVal.length; i++)
                partVal[i] = 0.0;
            //write and sort distances
            Iterator<Double> it = pots.keySet().iterator();
            double[] vals = new double[pots.size()];
            int c = 0;
            while (it.hasNext()) {
                vals[c] = it.next();
                c++;
            }
            Arrays.sort(vals);

            double all = 0; //funds already assigned to be forwarded
            for (int i = 0; i < vals.length; i++) {
                //start with node(s) at least distance
                Vector<Integer> vec = pots.get(vals[i]);
                while (vec.size() > 0) {
                    int node = vec.remove(rand.nextInt(vec.size()));
                    //forward all that still needs forwarding via this node if possible, otherwise: forward maximal value that can go via channel
                    double valNode = Math.min(rp.computePotential(cur, out[node]), curVal-all);
                    all = all + valNode;
                    partVal[node] = valNode;
                    if (all >= curVal) {
                        //stop if all funds are assigned for forwarding
                        break;
                    }
                }
                if (all >= curVal) {
                    break;
                }
            }
            return partVal;
        }
        else {
            //if(ratio<-0.1) return null;
            if (ratio >= 0)
                curVal = curVal;//*min(1,0.7+ratio*2);
                //else if(ratio>-0.1) curVal = 0;
            else curVal = sum;

            //sort nodes by potential (available funds)
            double[] partVal = new double[out.length];
            for (int i = 0; i < partVal.length; i++)
                partVal[i] = 0.0;
            Iterator<Double> it = pots.keySet().iterator();
            double[] vals = new double[pots.size()];
            int c = 0;
            while (it.hasNext()) {
                vals[c] = it.next();
                c++;
            }
            Arrays.sort(vals);
            double all = 0; //already assigned funds
            ArrayList<Integer> consumedNodes = new ArrayList<>();
            //iteratively assign funds to be forwarded to neighors
            /*for (int i = vals.length-1; i > -1; i--) {
                //start with nodes with highest funds to reduce splitting
                Vector<Integer> vec = pots.get(vals[i]);
                while (vec.size() > 0) {
                    int node = vec.remove(rand.nextInt(vec.size()));
                    //assign as many funds as possible to this node: all remaining funds if possible, otherwise maximum they can forward
                    double valNode = Math.min(rp.computePotential(cur, out[node]), curVal-all);
                    all = all + valNode;
                    partVal[node] = valNode;
                    if (all >= curVal) {
                        //if all funds assigned, stop
                        break;
                    }
                }
                if (all >= curVal) {
                    //if all funds assigned, stop
                    break;
                }
            }
            return partVal;*/
            for (int i = vals.length - 1; i > -1; i--) {
                //start with nodes with highest funds to reduce splitting
                Vector<Integer> vec = pots.get(vals[i]);
                while (vec.size() > 0) {
                    int node = vec.remove(rand.nextInt(vec.size()));
                    if (i > 0) {
                        double difference = vals[i] - vals[i - 1];
                        if (curVal - all > consumedNodes.size() * difference) {
                            for (int j = 0; j < consumedNodes.size(); j++) {
                                partVal[consumedNodes.get(j)] += difference;
                            }
                            all += difference * consumedNodes.size();
                        } else {
                            double addedFunds = (curVal - all) / consumedNodes.size();
                            for (int j = 0; j < consumedNodes.size(); j++) {
                                partVal[consumedNodes.get(j)] += addedFunds;
                            }
                            all = curVal;
                        }
                        consumedNodes.add(node);
                    } else {
                        consumedNodes.add(node);
                        while (vec.size() > 0) {
                            int temp = vec.remove(rand.nextInt(vec.size()));
                            consumedNodes.add(temp);
                        }
                        double addedFunds = (curVal - all) / consumedNodes.size();
                        for (int j = 0; j < consumedNodes.size(); j++) {
                            partVal[consumedNodes.get(j)] += addedFunds;
                        }
                        all = curVal;
                        break;
                    }


                    if (all >= curVal) {
                        //if all funds assigned, stop
                        break;
                    }
                }
                if (all >= curVal) {
                    double summ = 0.0;
                    for (int j = 0; j < consumedNodes.size() - 1; j++) {
                        summ += partVal[consumedNodes.get(j)];
                    }
                    partVal[consumedNodes.get(consumedNodes.size() - 1)] = curVal - summ;
                    //if all funds assigned, stop
                    break;
                }
            }
            return partVal;
        }
    }

}
