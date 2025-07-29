package tsp;

// Traveling Salesman Problem
//
// V4o: ordered (o) - simple ordering heuristic
//      bulk pruning by bound (4) - MST bounding heuristic;
//                                  bulk pruning relies on bound and ordering
//
// compile: javac -cp .. TSPV4o.java
// run:     java -cp .. tsp.TSPV4o PROBLEM_FILE [OPTIONS]

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashSet;
import yewpar.OptInstance;
import yewpar.Generator;
import yewpar.logger.TimeoutException;
import yewpar.logger.Logger;
import yewpar.logger.NoLogger;
import yewpar.logger.CountLogger;
import yewpar.logger.HistLogger;
import yewpar.logger.TracePredicate;
import yewpar.logger.Utils;
import tsp.TSPInstance;

public class TSPV4o extends OptInstance<TSPV4o.Node, Integer>
{
  final TSPInstance tsp;  // problem instance (dimension, distance matrix)

  /////////////////////////////////////////////////////////////////////////////
  // Search tree node: representation of current tour and remaining cities.
  // Cities are represented as non-negative integers < tsp.dim().
  // Instances of this class are immutable.
  static class Node
  {
    final int length;               // length of current tour (excl loop back)
    final int back;                 // loop back distance (last to first city)
    final int bound;                // lower bound on length of tour extensions
    final ArrayList<Integer> tour;  // current tour
    final HashSet<Integer> remain;  // remaining cities to add to tour

    // Invariants:
    // * The sets tour and remain are disjoint
    // * The sum of distances along the tour equals length
    // * If remain is empty (i.e. if tour is complete) length + back equals
    //   the sum of distances on a roundtrip tour

    Node(int length,
	 int back,
         int bound,
         ArrayList<Integer> tour,
         HashSet<Integer> remain) {
      this.length = length;
      this.back = back;
      this.bound = bound;
      this.tour = tour;
      this.remain = remain;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Lazy node generator: dictates how to traverse the search tree
  static class NodeGenerator implements Generator<Node>
  {
    // immutable fields
    final TSPInstance tsp;     // TSP instance
    final Node parent;         // NodeGen generates children of this parent node
    final int bound;           // lower bound on length of parent.tour exts
    final Integer[] candidates;  // candidates to extend parent.tour

    // mutable fields
    int k;                       // current index into array candidates

    NodeGenerator(TSPInstance tsp, Node p) {
      this.tsp = tsp;
      this.parent = p;
      this.k = p.remain.size();
      this.candidates = new Integer[this.k];
      int i = 0;
      for (int city : p.remain)
        this.candidates[i++] = city;
      // Sort candidates by decreasing distance from last city; that is,
      // the last elem of candidates should be closest to the last visited city
      int last = p.tour.get(p.tour.size() - 1);
      Arrays.sort(candidates,
                  (u, v) -> { return tsp.dist(v, last) - tsp.dist(u, last); });
      // Lower bound: weightMST(remaining cities) + loop back dist to first city
      int first = p.tour.get(0);
      bound = tsp.weightMST(p.remain) + tsp.dist(first, p.remain);
    }

    public int hasNext() { return k; }

    public Node next() {
      if (k == 0)
        return null;
      // get current first and last city
      int first = parent.tour.get(0);
      int last = parent.tour.get(parent.tour.size() - 1);
      // move index and get next city
      k--;
      int next = candidates[k];
      // update tour length and loop back distance
      int length = parent.length + tsp.dist(last, next);
      int back = tsp.dist(next, first);
      // copy parent tour and add next city
      ArrayList<Integer> tour = new ArrayList<>(parent.tour);
      tour.add(next);
      // copy parent.remain, but delete next city
      HashSet<Integer> remain = new HashSet<>(parent.remain);
      remain.remove(next);
      // construct the new Node
      return new Node(length, back, bound, tour, remain);
    }

    public Generator<Node> children(Node node) {
      return new NodeGenerator(tsp, node);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Objective function to be maximised: negation of roundtrip tour length
  public Integer objective(Node current) {
    if (current.remain.isEmpty())
      return -(current.length + current.back);
    else
      return Integer.MIN_VALUE;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Bulk pruning condition; relies on MST bound and ordering heuristic
  public int prune(Node current, Node incumbent) {
    if (incumbent.remain.isEmpty() &&
        current.length + current.bound >= incumbent.length + incumbent.back)
      return -1;
    return 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Boilerplate for setting up and running instances

  // private instance constructor
  private TSPV4o(Node root, Generator<Node> gen0, TSPInstance tsp) {
    super(root, gen0);
    this.tsp = tsp;
  }

  // static instance constructor method; returns null if tsp is null.
  public static TSPV4o mkTSP(TSPInstance tsp) {
    if (tsp == null)
      return null;
    ArrayList<Integer> tour = new ArrayList<>();
    tour.add(0);
    HashSet<Integer> remain = new HashSet<>();
    for (int city = 1; city < tsp.dim(); city++)
      remain.add(city);
    Node root = new Node(0, 0, 0, tour, remain);
    NodeGenerator gen0 = new NodeGenerator(tsp, root);
    return new TSPV4o(root, gen0, tsp);
  }

  // main function
  public static void main(String[] args) {
    try {
      TSPInstance tsp = TSPInstance.parse(args[0]);
      TSPV4o inst = mkTSP(tsp);
      if (inst == null)
        throw new Exception();
      System.out.print(args[0] + " ");
      TracePredicate tp = Utils.mkTracePredicate(args);
      Logger<Node> lg = new NoLogger<>();
      if (Utils.parseOptCountLogger(args)) { lg = new CountLogger<>(tp); }
      if (Utils.parseOptHistLogger(args))  { lg = new HistLogger<>(tp); }
      lg.setTimeout(Utils.parseOptTimeout(args));
      long t0 = System.nanoTime();
      Node x = inst.search(lg);
      long t1 = System.nanoTime();
      // System.out.print("Tour:");
      // for (int city : x.tour)
      //   System.out.print(" " + city);
      // System.out.println();
      System.out.print(" Length: " + -inst.objective(x));
      System.out.print(" Time: " + ((t1 - t0) / 1000000) + " ms \n");
    }
    catch (TimeoutException e) { System.out.println("\n Timeout"); }
    catch (Exception e) {
      System.out.println("Usage: java TSP PROBLEM_FILE [OPTIONS]");
    }
  }
}
