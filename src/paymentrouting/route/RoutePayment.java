
package paymentrouting.route;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.Map.Entry;

import gtna.data.Single;
import gtna.graph.Edge;
import gtna.graph.Edges;
import gtna.graph.Graph;
import gtna.graph.Node;
import gtna.io.DataWriter;
import gtna.metrics.Metric;
import gtna.networks.Network;
import gtna.util.Distribution;
import gtna.util.parameter.BooleanParameter;
import gtna.util.parameter.IntParameter;
import gtna.util.parameter.Parameter;
import gtna.util.parameter.StringParameter;
import paymentrouting.datasets.TransactionList;
import treeembedding.credit.CreditLinks;
import treeembedding.credit.Transaction;

import static java.lang.Math.abs;

/**
 * basic routing of payments, no concurrency but dynamics possible  
 * @author mephisto
 *
 */
public class RoutePayment extends Metric{
	//Parameters:
	protected int totalTime = 1000;
	protected int endTime = 1000;
	protected int delay = 10;
	protected double imbanlanceRate;
	protected double occupiedFunds = 0;
	protected double depletedChannels = 0;
	protected boolean intervalCount = false;
	protected boolean uselessCount = false;
	protected boolean boom = false;
	protected boolean flexibaleSize = false;
	protected boolean sourceRoute = true;
	protected int boomNumber = 100;
	protected double cRedundant = 0;
	protected double feeRate = 0.4;
	protected double sizeRate = 0.8;
	protected double totalFeeRate = 0.01;
	protected Random rand; //random seed
	protected boolean update; //are balances updated after payment or returned to original  
	protected Transaction[] transactions; //list of transactions 
	protected boolean log = false; //give detailed output in form of prints 
	protected PathSelection select; //splitting method  
	protected CreditLinks edgeweights; //the balances of the channels
	protected int tInterval = 200; //default length of an epoch (if you want to see success over time: averages taken for this number of transactions)
	protected int recompute_epoch; //do you recompute routing info periodically all tInterval? (if so, value < Integer.MAX_VALUE, which is default)
	protected int trials; //number of attempts payment is tried (didn't evaluate more than 1) 
	
	//Metrics: 
	protected Distribution hopDistribution; //number of hops of longest path (distribution)
	protected double avHops; //average number of hops of longest paths in a split payment 
	protected Distribution messageDistribution; //number of messages needed to route payment 
	protected double avMess; //average number of messages needed to route a pyment
	protected Distribution hopDistributionSucc; //number of hops of longest path (distribution) only considering successful payments 
	protected double avHopsSucc; //average number of hops for successful payments 
	protected Distribution messageDistributionSucc; //messages needed to route a payment for successful payments 
	protected double avMessSucc; //average number of messages for successful payments 
	protected Distribution trysDistribution; //number of attempts used until payment successful (or maximal if unsuccessful) 
	protected double success; //fraction of payments successful
	protected double successFirst; //fraction of payments successful in first try
	protected double[] succTime; //fraction of successful payments over time
	protected ArrayList<Integer>[] timeQueue; //store unfinished payments
	protected Vector<PartialPath>[] storedPPS; //store unfinished payments
	protected int dsts[];
	protected int srcs[];
	protected boolean flags[];
	protected double vals[];
	protected double oldVals[];
	protected int hopCount[];
	protected int messageCount[];
	protected int maxHopCount[];
	protected int lastSuc = 0;
	protected int lastFail = 0;
	protected int sourceList[]; //store unfinished payments
	protected int destList[]; //store unfinished payments
	protected double valList[]; //store unfinished payments
	protected List<List<int[]>> boomPaths = new ArrayList<>();
	public RoutePayment(PathSelection ps, int trials, boolean up,double redundancy) {
		this(ps,trials,up,Integer.MAX_VALUE,redundancy);
	}
	
    /**
     * basic constructor
     * @param ps
     * @param trials
     * @param up
     * @param epoch
     */
	public RoutePayment(PathSelection ps, int trials, boolean up, int epoch,double redundancy) {
		super("ROUTE_PAYMENT", new Parameter[]{new StringParameter("SELECTION", ps.getName()), new IntParameter("TRIALS", trials),
				new BooleanParameter("UPDATE", up), new StringParameter("DISTANCE", ps.getDist().name), 
				new IntParameter("EPOCH", epoch)});
		this.trials = trials;		
		this.update = up;
		this.select = ps; 
		this.recompute_epoch = epoch;
		this.cRedundant = redundancy;
	}
	
	/**
	 * constructor called by child classes needing more parameters 
	 * @param ps
	 * @param trials
	 * @param up
	 * @param epoch
	 * @param params
	 */
	public RoutePayment(PathSelection ps, int trials, boolean up, int epoch, Parameter[] params) {
		super("ROUTE_PAYMENT", extendParams(ps.getName(), trials, up, ps.getDist().name, epoch, params));
		this.trials = trials;		
		this.update = up;
		this.select = ps; 	
		this.recompute_epoch = epoch; 
	}
	
	/**
	 * constructor called by child classes needing more parameters that do not recompute info, i.e., this.recompute_epoch = Integer.MAX_VALUE
	 * @param ps
	 * @param trials
	 * @param up
	 * @param params
	 */
	public RoutePayment(PathSelection ps, int trials, boolean up, Parameter[] params) {
		this(ps, trials, up, Integer.MAX_VALUE, params); 		
	}
	
	public static Parameter[] extendParams(String selName, int trials, boolean up, String distName, int epoch, Parameter[] params) {
		Parameter[] nparams = new Parameter[params.length + 5];
		nparams[0] = new IntParameter("TRIALS", trials);
		nparams[1] = new BooleanParameter("UPDATE", up);
		nparams[2] = new StringParameter("DISTANCE", distName);
		nparams[3] = new StringParameter("SELECTION", selName);
		nparams[4] = new IntParameter("EPOCH", epoch);
		for (int i = 0; i < params.length; i++) {
			nparams[i+5] = params[i]; 
		}
		return nparams;
	}
	
	@Override
	public void computeData(Graph g, Network n, HashMap<String, Metric> m) {
		//init values
		//initialize weights
		sourceList = new int[300000];
		destList = new int[300000];
		valList = new double[300000];
		if(this.cRedundant>1)
		{
			double temp = this.cRedundant;
			double temp2 = (temp*100000%1000);
			temp -= temp2/100000;
			if(abs((temp*100)%10-1)<0.001) {
				this.boom = true;
				//System.out.println("boom");
				this.cRedundant-=0.01;
			}
			System.out.println("revoke protocol: "+this.cRedundant);

			computeData2(g,n,m);
			return;
		}
		else
		{
			this.cRedundant = 1.0000001;
			computeData2(g,n,m);
			return;
		}
	}
	
	@Override
	public boolean writeData(String folder) {
		boolean succ = true;
		succ &= DataWriter.writeWithIndex(this.messageDistribution.getDistribution(),
				this.key+"_MESSAGES", folder);
		succ &= DataWriter.writeWithIndex(this.messageDistributionSucc.getDistribution(),
				this.key+"_MESSAGES_SUCC", folder);
		succ &= DataWriter.writeWithIndex(this.hopDistribution.getDistribution(),
				this.key+"_HOPS", folder);
		succ &= DataWriter.writeWithIndex(this.hopDistributionSucc.getDistribution(),
				this.key+"_HOPS_SUCC", folder);
		succ &= DataWriter.writeWithIndex(this.trysDistribution.getDistribution(),
				this.key+"_TRYS", folder);
		succ &= DataWriter.writeWithIndex(this.succTime, this.key+"_SUCCESS_TEMPORAL", folder);
		
		return succ;
	}

	@Override
	public Single[] getSingles() {
		Single m_av = new Single(this.key + "_MES_AV", this.avMess);
		Single m_av_succ = new Single(this.key + "_MES_AV_SUCC", this.avMessSucc);
		Single h_av = new Single(this.key + "_HOPS_AV", this.avHops);
		Single h_av_succ = new Single(this.key + "_HOPS_AV_SUCC", this.avHopsSucc);
		
		Single s1 = new Single(this.key + "_SUCCESS_DIRECT", this.successFirst);
		Single s = new Single(this.key + "_SUCCESS", this.success);

		return new Single[]{m_av, m_av_succ, h_av, h_av_succ, s1, s};
	}
	

	
	protected long[] inc(long[] values, int index) {
		try {
			values[index]++;
			return values;
		} catch (ArrayIndexOutOfBoundsException e) {
			long[] valuesNew = new long[index + 1];
			System.arraycopy(values, 0, valuesNew, 0, values.length);
			valuesNew[index] = 1;
			return valuesNew;
		}
	}
	
	/**
	 * update the balances in edgeweights to balances in updateWeight 
	 * @param edgeweights
	 * @param updateWeight
	 */
	protected void weightUpdate(CreditLinks edgeweights, HashMap<Edge, Double> updateWeight){
		Iterator<Entry<Edge, Double>> it = updateWeight.entrySet().iterator();
		while (it.hasNext()){
			Entry<Edge, Double> entry = it.next();
			edgeweights.setWeight(entry.getKey(), entry.getValue());
		}
	}
	
	@Override
	/**
	 * need a graph that has channels and transactions to perform this metric 
	 */
	public boolean applicable(Graph g, Network n, HashMap<String, Metric> m) {
		return g.hasProperty("CREDIT_LINKS") && g.hasProperty("TRANSACTION_LIST");
	}
	
	/**
	 * randomly split val over r dimensions
	 * @param val
	 * @param r
	 * @param rand
	 * @return
	 */
	double[] splitRealities(double val, int r, Random rand) {
		double[] res = new double[r];
		for (int i = 0; i < r-1; i++) {
			res[i] = rand.nextDouble()*val;
		}
		res[res.length-1] = val;
		Arrays.sort(res);
		for (int i = r-1; i > 0; i--) {
			res[i] = res[i] -res[i-1];
		}
		return res;
	}
	
	/**
	 * merge all requests arriving at a node 
	 * @param unmerged: paths before merging
	 * @return
	 */
	protected Vector<PartialPath> merge(Vector<PartialPath> unmerged){
		Vector<PartialPath> vec = new Vector<PartialPath>();
		HashMap<Integer, HashSet<Integer>> dealtWith = new HashMap<Integer, HashSet<Integer>>(); //path per dimension 
		for (int i = 0; i < unmerged.size(); i++) {
			PartialPath p = unmerged.get(i); //path to merge (only with other path in same dimension) 
			int node = p.node;
			int r = p.reality;
			vec.add(new PartialPath(p.identify,p.id,node, p.val, p.pre,r,p.fee));
			HashSet<Integer> dealt = dealtWith.get(r);
			if (dealt == null) {
				//add new set for this dimension 
				dealt = new HashSet<Integer>();
				dealtWith.put(r, dealt);
			}
/*			if (!dealt.contains(node)) {
				dealt.add(node);
				Vector<Integer> contained = p.pre;
				double valSum = p.val;
				//merge with any other paths that arrived at same node
				for (int j = i+1; j < unmerged.size(); j++) {
					PartialPath m = unmerged.get(j);
					if (m.node == node && m.reality == r) {
						//add all nodes to path so that they are excluded during routing
						//(might result in duplicates, but this is good enough as paths are short)
						Vector<Integer> toAdd = m.pre;
						//start at 1, because 0 is the same for all paths
						for (int l = 1; l < toAdd.size(); l++) {
							int cur = toAdd.get(l);
							if (!contained.contains(cur)) {
								contained.add(cur);
							}
						}
						valSum = valSum + m.val;
						if (log) {
							System.out.println("Merge at " + node + " new val " + valSum);
						}
					}
				}
				vec.add(new PartialPath(node, valSum, contained,r));
			}*/
		}
		return vec; 
	}
	
	/**
	 * get available funds of link (s,t) 
	 * for atomic non-concurrent payment: partial payment goes through iff all payments go through,
	 * hence consider other operations on link as if they succeed
	 * OVERRIDE FOR OTHER PAYMENTS  
	 * @param s
	 * @param t
	 * @return
	 */
	public double computePotential(int s, int t) {
		return this.edgeweights.getPot(s, t);
	}
	
	/**
	 * return total capacity of a channel 
	 * @param s
	 * @param t
	 * @return
	 */
	public double getTotalCapacity(int s, int t) {
		return this.edgeweights.getTotalCapacity(s, t); 
	}

	public void computeData2(Graph g, Network n, HashMap<String, Metric> m) {
		//init values
		rand = new Random();
		this.select.initRoutingInfo(g, rand);
		edgeweights = (CreditLinks) g.getProperty("CREDIT_LINKS");
		BufferedReader in = null;
		Node[] nodes = g.getNodes();
		if(nodes.length==18081) {
			if(((this.cRedundant*1000)%10)/10>0&&flexibaleSize) {
				this.sizeRate = ((this.cRedundant * 1000) % 10) / 10;
				this.cRedundant -= this.sizeRate/100;
				//System.out.println(this.sizeRate);
				//System.out.println(this.cRedundant);
			}
			try {
				in = new BufferedReader(new FileReader("lightning/ln_capacity.txt"));
				String str;
				while ((str = in.readLine()) != null) {
					//System.out.println(str);
					String[] ss = str.split("\\s+");
					//System.out.println(ss);
					int s1 = Integer.parseInt(ss[0]);
					int s2 = Integer.parseInt(ss[1]);
					double s3 = Double.parseDouble(ss[2]);
					//System.out.println("ha: "+s1+' '+s2+' '+s3);
					Edge e = edgeweights.makeEdge(s1, s2);
					edgeweights.setWeight(e, new double[]{0, s3 / 2, s3});
					//System.out.println(" "+s1+' '+s1+' '+s3/2);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		else
		{
			Edges allEdges = g.getEdges();
			for (int iii = 0; iii < allEdges.size(); iii++) {
				Edge ee = allEdges.getEdges().get(iii);
				int s1 = ee.getSrc();
				int s2 = ee.getDst();
				sourceList[iii] = s1;
				destList[iii] = s2;
				valList[iii] = this.computePotential(s1, s2);
			}
		}
		//System.out.println("size rate is: "+this.sizeRate);
		HashMap<Edge, Double> originalAll = new HashMap<Edge, Double>();
		this.transactions = ((TransactionList) g.getProperty("TRANSACTION_LIST")).getTransactions();
		int count = this.transactions.length;

		this.avHops = 0;
		this.avHopsSucc = 0;
		this.avMess = 0;
		this.avMessSucc = 0;
		this.successFirst = 0;
		this.success = 0;
		this.timeQueue = new ArrayList[totalTime*2+3000];
		for (int i = 0; i < totalTime*2+3000; i++)
			this.timeQueue[i] = new ArrayList<>();
		this.storedPPS = new Vector[count*2];
		long[] trys = new long[2];
		long[] path = new long[2];
		long[] pathSucc = new long[2];
		long[] mes = new long[2];
		long[] mesSucc = new long[2];
		oldVals = new double[count*2];
		double successSum = 0;
		dsts = new int[count*2];
		srcs = new int[count*2];
		flags = new boolean[count*2];
		hopCount= new int[count*2];
		messageCount= new int[count*2];
		maxHopCount= new int[count*2];
		vals = new double[count*2];
		int uselessCnt = 0;
		SybilProofFee feePolicy = new SybilProofFee(1-feeRate);
		int len = this.transactions.length / this.tInterval;
		int rest = this.transactions.length % this.tInterval;
		if (rest == 0) {
			this.succTime = new double[len];
		} else {
			this.succTime = new double[len + 1];
		}
		int slot = 0;

		//iterate over transactions
		for (int i = 0; i < this.transactions.length; i++) {
			Transaction tr = this.transactions[i];
			boomPaths.add(null);
			int src = tr.getSrc();
			int dst = tr.getDst();
			dsts[i] = dst;
			srcs[i] = src;
			Node[] nodess = g.getNodes();
			int[] out = nodess[src].getOutgoingEdges();
			double tempSum = 0;
			for (int k = 0; k < out.length; k++) {
				if (this.select.dist.isCloser(out[k], src, dst, 1)) {
					double pot = this.computePotential(src, out[k]);
					tempSum+=pot;
				}
			}
			double val = tr.getVal();
			if(nodes.length==18081)
				val = tempSum*sizeRate;
			vals[i] = val;
			//add redundancy to payments and store the original value
			val = val*cRedundant;
			oldVals[i] = val;
			boolean s = true; //successful, reset when failure encountered
			flags[i] = s;
			hopCount[i] = 0;
			messageCount[i] = 0;
			int maxhops = this.select.getDist().getTimeLock(src, dst); //maximal length of path
			maxHopCount[i] = maxhops;
			if(tempSum<vals[i]*(1+totalFeeRate))
			{
				uselessCnt++;
				continue;
			}
			out = nodess[dst].getOutgoingEdges();
			tempSum = 0;
			for (int k = 0; k < out.length; k++) {
				{
					double pot = this.computePotential(out[k],dst);
					tempSum+=pot;
				}
			}
			if(tempSum<vals[i]*(1+totalFeeRate))
			{
				uselessCnt++;
				continue;
			}
			Vector<PartialPath> pps = new Vector<PartialPath>();
			//some routing algorithm split over multiple dimensions in the beginning (!= splitting during routing)
			double[] splitVal = this.splitRealities(val, select.getDist().startR, rand);
			if(!this.sourceRoute) {
				for (int a = 0; a < select.getDist().startR; a++) {
					pps.add(new PartialPath(src, splitVal[a], new Vector<Integer>(), a, splitVal[a] * totalFeeRate));
				}
			}
			else
			{
				boomPaths.set(i,getEdgeDisjointPaths(g,src,dst,25));
				if(i%1000==0)
					System.out.println("max flows "+ i);
				Random rn = new Random();
				double total = 0.0;
				for (int a = 0; a < select.getDist().startR; a++) {
					double pakectSize =  splitVal[a]/this.boomNumber;
					for(int k=0;k<this.boomNumber;k++) {
						int tempId = -abs(k)%(boomPaths.get(i).size());
						total+=pakectSize;
						pps.add(new PartialPath(k,tempId,src, pakectSize, new Vector<Integer>(), a, pakectSize * totalFeeRate));
					}
					//System.out.println("size is :"+total);
				}
				//System.out.println("henhen "+i);
			}
			this.storedPPS[i] = pps;
			if(this.transactions.length<totalTime)
				this.timeQueue[totalTime/this.transactions.length*i].add(i);
			else this.timeQueue[i % totalTime].add(i);
		}
		//execute payments by time order
		for (int i = 0; i < endTime; i++) {
			//if(i%100==0)
			//	System.out.println(i+" seconds");
			//iterate over payments to be handled in one second
			for (int ii = 0; ii < this.timeQueue[i].size(); ii++) {
				int id = timeQueue[i].get(ii);
				Vector<PartialPath> pps = this.storedPPS[id];

				//some routing algorithm split over multiple dimensions in the beginning (!= splitting during routing)
				boolean[] excluded = new boolean[nodes.length];

				//HashMap<Edge, Double> originalWeight = new HashMap<Edge, Double>(); //updated weights
				int revoked = 0;
				//while current set of nodes is not empty
				if (!pps.isEmpty() && hopCount[id] < maxHopCount[id]) {
					if (log) {
						System.out.println("Hop " + hopCount[id] + " with " + pps.size() + " links ");
					}
					Vector<PartialPath> next = new Vector<PartialPath>();
					//iterate over set of current set of nodes
					double sumVal = 0.0;
					for (int j = 0; j < pps.size(); j++) {
						PartialPath pp = pps.get(j);
						int ppid = pp.id;;
						//System.out.println(pp.val+" "+pp.fee);
						int cur = pp.node;
						//exclude nodes already on the path
						int pre = -1;
						Vector<Integer> past = pp.pre;
						if (past.size() > 0) {
							pre = past.get(past.size() - 1);
						}
						for (int l = 0; l < past.size(); l++) {
							excluded[past.get(l)] = true;
						}

						if (log) System.out.println("Routing at cur " + cur);
						//getNextVals -> distribution of payment value over neighbors
						double feeRatio = feePolicy.charge(pp.fee)/(pp.val+feePolicy.charge(pp.fee));
						double[] partVals;
						if(this.sourceRoute)
						{
							int[] outEdge = nodes[cur].getOutgoingEdges();
							partVals = new double[outEdge.length];
							for(int iii=0;iii<outEdge.length;iii++)
								partVals[iii] = 0.0;
							if(pp.id<0)
							{
								pp.id = -pp.id;
								int tempId = pp.id;
								//System.out.println(pp.id);
								for(int iid=0;iid<boomPaths.get(id).size();iid++)
								{
									tempId = (pp.id+iid)%(boomPaths.get(id).size());
									int[] tempPath = boomPaths.get(id).get(tempId);
									//System.out.println(cur+" "+tempPath[0]+" "+tempPath[1]);
									if(this.computePotential(cur, tempPath[1])>=pp.val+pp.fee)
										pp.id = tempId;
								}
								ppid = pp.id;
								/*String pppp = "";
								for(int cntp = 0;cntp<boomPaths.get(id).get(pp.id).length;cntp++)
									pppp+=(boomPaths.get(id).get(pp.id)[cntp]+" ");
								System.out.println(pp.identify+" "+id+" "+tempId+" : "+pppp);*/
								//System.out.println("owari");
							}
							int[] boomPath = boomPaths.get(id).get(pp.id);
							int nextNode = -1;
							int found = 0;
							for(int l=0;l<boomPath.length;l++)
							{
								if(boomPath[l] == cur)
								{
									found = 1;
									nextNode = boomPath[l+1];
									double capa = this.computePotential(cur, nextNode);
									if(capa>=pp.val+feePolicy.charge(pp.fee))
									{
										for(int ll = 0;ll<outEdge.length;ll++)
										{
											if(outEdge[ll] == nextNode) {
												partVals[ll] = pp.val + feePolicy.charge(pp.fee);
												break;
											}
										}
									}
									else
									{
										//System.out.println(pp.identify+" "+id+" "+ppid+" "+srcs[id]+" "+cur+" "+nextNode+" insufficient fund "+capa+" "+(pp.val+feePolicy.charge(pp.fee)));
									}
									break;
								}
							}
							if(found==0)
								System.out.println("???? "+cur);
						}
						else
						{
							partVals = this.select.getNextsVals(g, cur, dsts[id],
									pre, excluded, this, pp.val+feePolicy.charge(pp.fee), rand, pp.reality);
						}
						double tempSum = 0.0;
						//reset excluded for future use
						for (int l = 0; l < past.size(); l++) {
							excluded[past.get(l)] = false;
						}
						//add neighbors that are not the dest to new set of current nodes
						if (partVals != null) {
							//past.add(cur);
							for(int l=0;l<partVals.length;l++)
								tempSum+=partVals[l];
							oldVals[id] -= (pp.val - tempSum);
							if(oldVals[id]<vals[id])
							{
								//System.out.println("nmd");
								//failure to find nodes to route to
								revoked = 1;
								flags[id] = false;
								//break;
								//System.out.println("haha");
								for (int hh = j+1; hh < pps.size(); hh++) {
									PartialPath tempp = pps.get(hh);
									Vector<Integer> tempPAST = (Vector<Integer>) tempp.pre.clone();
									tempPAST.add(tempp.node);
									revoke(tempp.pre,tempp.val,tempp.fee);
								}
								for(int hh = 0;hh<next.size();hh++)
								{
									PartialPath tempp = next.get(hh);
									Vector<Integer> tempPAST = (Vector<Integer>) tempp.pre.clone();
									tempPAST.add(tempp.node);
									revoke(tempp.pre,tempp.val,tempp.fee);
								}
								//recovery to original weights
								break;
							}
							if(tempSum<pp.val) {
								//System.out.println("haha");
								Vector<Integer> tempPAST = (Vector<Integer>) past.clone();
								tempPAST.add(cur);
								revoke(tempPAST, pp.val - tempSum,pp.fee*(pp.val-tempSum)/pp.val);
							}
							sumVal+=tempSum;
							int[] out = nodes[cur].getOutgoingEdges();
							int zeros = 0; //delay -> stay at same node (happens as part of attacks, ignore if not in attack scenario)
							for (int k = 0; k < partVals.length; k++) {
								if (partVals[k] > 0) {
									messageCount[id]++;
									//update vals
									Edge e = edgeweights.makeEdge(cur, out[k]);
									double w = edgeweights.getWeight(e);
									/*
									if (!originalWeight.containsKey(e)) {
										originalWeight.put(e, w); //store balance before this payment if later reset due to, e.g., failure
									}
									*/
									if (!originalAll.containsKey(e)) {
										originalAll.put(e, w); //store original balance before this execution (for other runs with different parameters)
									}
									//System.out.println(cur+" "+out[k]+" "+partVals[k]+" "+this.computePotential(cur,out[k]));
									edgeweights.setWeight(cur, out[k], partVals[k]);//set to new balance
									//System.out.println("after: "+cur+" "+out[k]+" "+partVals[k]+" "+this.computePotential(cur,out[k]));
									//System.out.println(srcs[id]+" to "+dsts[id]+" update "+cur+" "+out[k]);
									if (out[k] != dsts[id]) {
										Vector<Integer> tempPAST = (Vector<Integer>) past.clone();
										tempPAST.add(cur);
										//System.out.println("added: "+pp.identify);
										next.add(new PartialPath(pp.identify,pp.id,out[k], partVals[k]*(1-feeRatio),
												tempPAST, pp.reality,partVals[k]*feeRatio)); //add new intermediary to path
									}
									else
									{
										Vector<Integer> tempPAST = (Vector<Integer>) past.clone();
										tempPAST.add(cur);
										tempPAST.add(dsts[id]);
										if(partVals[k]>vals[id])
										{
											revoke(tempPAST,partVals[k]-vals[id],(partVals[k]-vals[id])*feeRatio);
											vals[id] = 0;
										}
										else vals[id]-=partVals[k];
									}
									if (log) {
										System.out.println("add link (" + cur + "," + out[k] + ") with val " + partVals[k]);
									}
								} else {
									zeros++; //node performs an attack by waiting until it forwards (ignore if not in attack scenario)
								}
							}
							if (zeros == partVals.length) {
								//stay at node itself
								//next.add(new PartialPath(pp.identify,ppid,cur, pp.val,
									//	(Vector<Integer>) past.clone(), pp.reality,pp.fee));
							}
						} else {
							//failure to find nodes to route to
							revoked = 1;
							flags[id] = false;
							//break;
							//System.out.println("haha");
							for (int hh = j+1; hh < pps.size(); hh++) {
								PartialPath tempp = pps.get(hh);
								Vector<Integer> tempPAST = (Vector<Integer>) tempp.pre.clone();
								tempPAST.add(tempp.node);
								revoke(tempp.pre,tempp.val,tempp.fee);
							}
							for(int hh = 0;hh<next.size();hh++)
							{
								PartialPath tempp = next.get(hh);
								Vector<Integer> tempPAST = (Vector<Integer>) tempp.pre.clone();
								tempPAST.add(tempp.node);
								revoke(tempp.pre,tempp.val,tempp.fee);
							}
							//recovery to original weights
							break;
						}
					}
					if(flags[id])
						pps = this.merge(next); //merge paths: if the same payment arrived at a node via two paths: merge into one
					hopCount[id]++; //increase hops

					//revoke too much, impossible to complete, so revoke all
					if(revoked==0&&vals[id]-sumVal>0.00001) {
						//System.out.println("???");
						flags[id] = false;
						revoked = 1;
						//System.out.println("nmd");
						for(int l=0;l< pps.size();l++)
						{
							for(int ll=0;ll<pps.size();ll++) {
								revoke(pps.get(ll).pre, pps.get(ll).val,pps.get(ll).fee);
							}
						}
					}
					else if(flags[id])
					{
						this.storedPPS[id] = pps;
						this.timeQueue[i+delay].add(id);
					}

					if (hopCount[id] == maxHopCount[id] && !pps.isEmpty()) {
						//System.out.println("ad");
						flags[id] = false;
					}
					if(flags[id]==false||pps.isEmpty())
					{
						this.select.clear(); //clear any information related to finished payment
						if (flags[id]==false) {
							hopCount[id]--;
							//System.out.println("nmd");
						} else {
							if (!this.update) {
								//return credit links to original state
								//this.weightUpdate(edgeweights, originalWeight); deprecated since concurrent payments exist
							}
							//update stats for this transaction
							//System.out.println(vals[id]+" "+sumVal);
							pathSucc = inc(pathSucc, hopCount[id]);
							mesSucc = inc(mesSucc, messageCount[id]);
							successSum += oldVals[id];
							this.succTime[slot]++;
							this.success++;
							//if (t == 0)
							{
								this.successFirst++;
							}
							//trys = inc(trys, t);
							if (log) {
								System.out.println("Success");
							}
						}
						path = inc(path, hopCount[id]);
						mes = inc(mes, messageCount[id]);
					}
					/*if ((i + 1) % this.tInterval == 0) {
						this.succTime[slot] = this.succTime[slot] / this.tInterval;
						slot++;
					}*/
				}

				//recompute routing info, e.g., spanning trees
				if (this.recompute_epoch != Integer.MAX_VALUE && (i + 1) % this.recompute_epoch == 0) {
					this.select.initRoutingInfo(g, rand);
				}
			}
			timeQueue[i].clear();
			//System.out.println("");
			if(i%tInterval==tInterval-1&&intervalCount==true&&i<totalTime) {
				int cnt = 0;
				for (int tt = 0; tt < this.transactions.length; tt++) {
					if (flags[tt] == false)
						cnt++;
				}
				System.out.println((this.success - lastSuc) / (cnt + this.success - lastSuc - lastFail));
				//System.out.println();
				lastSuc = (int) this.success;
				lastFail = cnt;
				if (nodes.length == 18081) {
					double ccnt = 0;
					try {
						depletedChannels = 0;
						imbanlanceRate = 0.0;
						double imbalance = 0.0;
						double totalCapacity = 0.0;
						in = new BufferedReader(new FileReader("lightning/ln_capacity.txt"));
						String str;
						while ((str = in.readLine()) != null) {
							//System.out.println(str);
							ccnt++;
							String[] ss = str.split("\\s+");
							//System.out.println(ss);
							int s1 = Integer.parseInt(ss[0]);
							int s2 = Integer.parseInt(ss[1]);
							double s3 = Double.parseDouble(ss[2]);
							//System.out.println("ha: "+s1+' '+s2+' '+s3);
							Edge e = edgeweights.makeEdge(s1, s2);
							double currentCapacity = this.computePotential(s1, s2);
							if (currentCapacity < 100)
								depletedChannels++;
							imbalance += abs(currentCapacity - s3/2);
							totalCapacity += s3/2;
							//edgeweights.setWeight(e, new double[]{0, s3 / 2, s3});
							//System.out.println(" "+s1+' '+s1+' '+s3/2);
						}
						imbanlanceRate = imbalance / totalCapacity;
					} catch (Exception e) {
						e.printStackTrace();
					}
					System.out.println("imbalance rate: " + imbanlanceRate);
					System.out.println("depleted channels: " + depletedChannels);
					occupiedFunds = 0;
					for (int tt = i + 1; tt < i + delay + 10; tt++) {
						//System.out.println(this.timeQueue[i].size());
						for (int ttt = 0; ttt < this.timeQueue[tt].size(); ttt++) {
							int id = timeQueue[tt].get(ttt);
							Vector<PartialPath> pps = this.storedPPS[id];
							if (!pps.isEmpty()) {
								for (int j = 0; j < pps.size(); j++) {
									PartialPath pp = pps.get(j);
									occupiedFunds += pp.pre.size() * pp.val;
									//System.out.println("asd "+pp.val+" "+pp.pre.size());
								}
							}
						}
					}
					System.out.println("occupied: " + occupiedFunds);
				}
				else
				{
					depletedChannels = 0;
					imbanlanceRate = 0.0;
					double imbalance = 0.0;
					double totalCapacity = 0.0;
					double hahah = 0.0;
					for(int iii=0; iii<g.getEdges().size();iii++) {
						int s1 = sourceList[iii];
						int s2 = destList[iii];
						double currentCapacity = this.computePotential(s1, s2);
						imbalance += abs(currentCapacity - valList[iii]);
						totalCapacity += valList[iii];
						hahah+=currentCapacity;
						//edgeweights.setWeight(e, new double[]{0, s3 / 2, s3});
						//System.out.println(" "+s1+' '+s1+' '+s3/2);
					}
					imbanlanceRate = imbalance / totalCapacity;
					System.out.println("imbalance rate: " + imbanlanceRate);
					System.out.println("depleted channels: " + depletedChannels);
					occupiedFunds = 0;
					for (int tt = i + 1; tt < i + delay + 10; tt++) {
						//System.out.println(this.timeQueue[i].size());
						for (int ttt = 0; ttt < this.timeQueue[tt].size(); ttt++) {
							int id = timeQueue[tt].get(ttt);
							Vector<PartialPath> pps = this.storedPPS[id];
							if (!pps.isEmpty()) {
								for (int j = 0; j < pps.size(); j++) {
									PartialPath pp = pps.get(j);
									occupiedFunds += pp.pre.size() * pp.val;
									//System.out.println("asd "+pp.val+" "+pp.pre.size());
								}
							}
						}
					}
					System.out.println("occupied: " + occupiedFunds);
				}
			}
		}

		//compute final stats
		this.hopDistribution = new Distribution(path,count);
		this.messageDistribution = new Distribution(mes,count);
		this.hopDistributionSucc = new Distribution(pathSucc,(int)this.success);
		this.messageDistributionSucc = new Distribution(mesSucc,(int)this.success);
		this.trysDistribution = new Distribution(trys,count);
		this.avHops = this.hopDistribution.getAverage();
		this.avHopsSucc = this.hopDistributionSucc.getAverage();
		this.avMess = this.messageDistribution.getAverage();
		this.avMessSucc = this.messageDistributionSucc.getAverage();
		int fcnt = 0;
		for (int tt = 0; tt < this.transactions.length; tt++) {
			if (flags[tt] == false)
				fcnt++;
		}
		this.success = this.success/(this.success+fcnt);
		System.out.println(success);
		if(uselessCount)
			System.out.println(uselessCnt);
		this.successFirst = this.successFirst/this.transactions.length;
		if (rest > 0) {
			this.succTime[this.succTime.length-1] = this.succTime[this.succTime.length-1]/rest;
		}

		//reset weights for further routing algorithms evaluated
		this.weightUpdate(edgeweights, originalAll);
	}

/*	public void revoke(Vector<Integer> pre,double val) {
		String str = "";
		for (int hhh = 0; hhh < pre.size(); hhh++)
			str += " " + pre.get(hhh);
		//System.out.println(str);
		for (int hhh = pre.size() - 2; hhh >= 0; hhh--) {
			edgeweights.setWeight(pre.get(hhh + 1), pre.get(hhh), val);//set to new balance
		}
	}*/
	public void revoke(Vector<Integer> pre,double val,double fee) {
		String str = "";
		for (int hhh = 0; hhh < pre.size(); hhh++)
			str += " " + pre.get(hhh);
		//System.out.println(str);
		for (int hhh = pre.size() - 2; hhh >= 0; hhh--) {
			edgeweights.setWeight(pre.get(hhh + 1), pre.get(hhh), val+fee);//set to new balance
			fee = fee/(1-feeRate);
		}
	}
	public List<int[]> getEdgeDisjointPaths(Graph g, int src, int dst, int k) {
		if(src==dst)return Collections.singletonList(new int[] {src});

		Map<Edge, Integer> flow = new HashMap<>();
		Map<Edge, Integer> capacity = new HashMap<>();
		for (Edge e : g.getEdges().getEdges()){
			flow.put(e, 0);
			capacity.put(e, 1);
			Edge r = new Edge(e.getDst(), e.getSrc());
			flow.put(r, 0);
			capacity.put(r, 1);
		}
		double totalflow = 0;
		int[] respath;
		while (totalflow < k && (respath = findResidualFlow(flow, capacity, g.getNodes(), src, dst)) != null){
			//pot flow along this path
			int min = Integer.MAX_VALUE;
			for (int i = 0; i < respath.length-1; i++){
				Edge e = new Edge(respath[i], respath[i+1]);
				int a = capacity.get(e) - flow.get(e);
				if (a < min){
					min = a;
				}
			}
			//update flows
			totalflow = totalflow + min;
			for (int i = 0; i < respath.length-1; i++){
				int n1 = respath[i];
				int n2 = respath[i+1];
				Edge e = new Edge(n1, n2);
				Edge r = new Edge(n2, n1);
				flow.put(e, flow.get(e) + min);   // push flow
				flow.put(r, capacity.get(e) - flow.get(e));   // un-push residual
			}
		}

		List<int[]> paths = new ArrayList<>();
		for (int i = 0; i < k; i++) {
			List<Integer> p = new ArrayList<>();
			p.add(src);
			int curr = src;

			while (curr != dst){
				boolean stuck = true;
				for (int next : g.getNodes()[curr].getOutgoingEdges()){
					Edge e = new Edge(curr, next);
					if (flow.get(e) > 0){
						flow.put(e, 0);   // do not reuse edge
						curr = next;
						stuck = false;
						p.add(curr);
						break;
					}
				}
				if (stuck) break;
			}
			if (curr == dst){   // move path from list to array
				int[] arrpath = new int[p.size()];
				int j = 0;
				for (int n : p)
					arrpath[j++] = n;
				paths.add(arrpath);
				p.clear();
			} else break;
		}
		return paths;
	}


	public int[] findResidualFlow(
			Map<Edge, Integer> flow,
			Map<Edge, Integer> capacity,
			Node[] nodes, int src, int dst){

		int[][] pre = new int[nodes.length][2];
		for (int i = 0; i < pre.length; i++){
			pre[i][0] = -1;
		}
		Queue<Integer> q = new LinkedList<Integer>();
		q.add(src);
		pre[src][0] = -2;
		while (!q.isEmpty()){
			int n1 = q.poll();
			int[] out = nodes[n1].getOutgoingEdges();
			for (int n: out){
				Edge e = new Edge(n1, n);
				if (pre[n][0] == -1 && capacity.get(e) > flow.get(e)){
					pre[n][0] = n1;
					pre[n][1] = pre[n1][1]+1;
					if (n == dst){
						int[] respath = new int[pre[n][1]+1];
						while (n != -2){
							respath[pre[n][1]] = n;
							n = pre[n][0];
						}
						return respath;
					}
					q.add(n);
				}

			}
		}
		return null;
	}
}




