package paymentrouting.route;

import java.util.Vector;

/**
 * info about path between source and intermediary 
 * @param n
 * @param val2
 * @param p
 * @param r
 */

public class PartialPath {
        public int node;
		public double val;
		public Vector<Integer> pre;
		public int reality;
		public int id;
		public double fee;
		public int identify;
		
		public PartialPath(int n, double val2, Vector<Integer> p, int r,double fee) {
			this.node = n;
			this.val = val2;
			this.pre = p;
			this.reality = r;
			this.fee = fee;
			this.id = 0;
			this.identify = 0;
		}
		public PartialPath(int id,int n, double val2, Vector<Integer> p, int r,double fee) {
			this.id = id;
			this.node = n;
			this.val = val2;
			this.pre = p;
			this.reality = r;
			this.fee = fee;
			this.identify = 0;
		}
	public PartialPath(int identify,int id,int n, double val2, Vector<Integer> p, int r,double fee) {
		this.id = id;
		this.node = n;
		this.val = val2;
		this.pre = p;
		this.reality = r;
		this.fee = fee;
		this.identify = identify;
	}
	
}
