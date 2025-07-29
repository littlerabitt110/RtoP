package tsp;

// TSP instance, represented as square matrix of integer distances

import java.io.File;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Scanner;

public class TSPInstance
{
  private int dim;    // dimension of square distance matrix
  private int[][] d;  // 2d array of distances (non-negative, symmetric)
  private ArrayList<Edge> edges;  // all edges, sorted by increasing distance

  // Constructs a TSP instance with dim cities from the given distance matrix,
  // presented linearly row by row;
  // requires dim > 0 && data.size >= dim^2.
  private TSPInstance(int dim, Collection<Integer> data) {
    // Set dimension
    this.dim = dim;
    // Set up distances
    this.d = new int[dim][dim];
    Iterator<Integer> it = data.iterator();
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        this.d[i][j] = it.next();
    // Set up and sort edge list by ascending distance
    this.edges = new ArrayList<>();
    for (int i = 0; i < dim; i++)
      for (int j = i + 1; j < dim; j++)
        this.edges.add(new Edge(i, j, this.d[i][j]));
    this.edges.sort( (e1, e2) -> { return e1.weight - e2.weight; } );
  }

  public int dim() { return dim; }

  // Returns the distance between two cities;
  // requires 0 <= city1, city2 < dim.
  public int dist(int city1, int city2) { return d[city1][city2]; }

  // Returns the minimum distance between city1 and one of the other cities;
  // returns -1 if collection cities is emtpy;
  // requires 0 <= city1 < dim && 0 <= city < dim for (city : cities).
  public int dist(int city1, Collection<Integer> cities) {
    if (cities.isEmpty())
      return -1;
    int x = Integer.MAX_VALUE;
    for (int city2 : cities)
      if (d[city1][city2] < x)
        x = d[city1][city2];
    return x;
  }

  // Returns true if this TSP instance is well-formed, i.e. all non-diagonal
  // entries of the distance matrix are positive, and the matrix is symmetric
  // with a zero diagonal.
  public boolean isWF() {
    if (dim == 0) {
      System.err.println("TSPInstance.isWF: dim == 0");
      return false;
    }
    if (dim != d.length) {
      System.err.println("TSPInstance.isWF: dim != d.length");
      return false;
    }
    for (int i = 0; i < dim; i++)
      if (dim != d[i].length) {
        System.err.println("TSPInstance.isWF: dim != d[" + i + "].length");
        return false;
      }
    for (int i = 0; i < dim; i++)
      if (d[i][i] != 0) {
        System.err.println("TSPInstance.isWF: d[" + i + "," + i + "] != 0");
        return false;
      }
    for (int i = 0; i < dim; i++)
      for (int j = i + 1; j < dim; j++) {
        if (d[i][j] <= 0) {
          System.err.println("TSPInstance.isWF: d[" + i + "," + j + "] <= 0");
          return false;
        }
        if (d[i][j] != d[j][i]) {
          System.err.println("TSPInstance.isWF: d[" + i + "," + j + "] " +
                             "!= d[" + j + "," + i + "]");
          return false;
        }
      }
    return true;
  }

  // Returns true if this TSP instance is a metric TSP, i.e. satisfies
  // the triangle inequality.
  public boolean isMetric() {
    for (int i = 0; i < dim; i++)
      for (int j = i + 1; j < dim; j++)
        for (int k = j + 1; k < dim; k++) {
          if (d[i][k] > d[i][j] + d[j][k])
            return false;
          if (d[i][j] > d[i][k] + d[k][j])
            return false;
          if (d[j][k] > d[j][i] + d[i][k])
            return false;
        }
    return true;
  }

  // Prints triples of cities that violate triangle inequality.
  public void nonMetric() {
    for (int i = 0; i < dim; i++)
      for (int j = i + 1; j < dim; j++)
        for (int k = j + 1; k < dim; k++) {
          int d_ij = d[i][j];
          int d_jk = d[j][k];
          int d_ik = d[i][k];
          if (d_ik > d_ij + d_jk || d_ij > d_ik + d_jk || d_jk > d_ij + d_ik)
            System.err.println("TSPInstance.nonMetric: triple " +
                               i + "," + j + "," + k +
                               " violates triangle inequality");
        }
  }

  // Constructs a TSP instnce from a file in the following format:
  // * the first line is the dimension, i.e. the number of cities;
  // * the following lines specify the distance matrix in row-major format;
  // * extranous, blank, or comment lines (starting with #) are ignored.
  // Returns null if the file can't be read or parsed.
  public static TSPInstance parse(String filename) {
    try {
      Scanner sc = new Scanner(new File(filename));
      int dim = -1;
      ArrayList<Integer> data = new ArrayList<>();
      while (sc.hasNext()) {
        // skip comment lines
        if (sc.hasNext("#.*")) {
          sc.nextLine();
          continue;
        }
        // read next integer (dimension, or an entry of the distance matrix)
        if (dim < 0) {
          dim = sc.nextInt();      // read dimension
        } else {
          data.add(sc.nextInt());  // read distance
        }
      }
      // sanity checks: dim negative and enough matrix entries read
      if (dim <= 0) {
        System.err.println("TSPInstance.parse: dim <= 0");
        return null;
      }
      if (data.size() < dim * dim) {
        System.err.println("TSPInstance.parse: not enough entries");
        return null;
      }
      TSPInstance tsp = new TSPInstance(dim, data);
      if (!tsp.isWF()) {
        System.err.println("TSPInstance.parse: not a distance matrix");
        return null;
      }
      if (!tsp.isMetric()) {
        System.err.println("TSPInstance.parse: warning - not a metric TSP");
	tsp.nonMetric();  // print triples that violate triangle inequality
      }
      return tsp;
    }
    catch (Exception e) {
      System.err.println("TSPInstance.parse: scanner error");
      return null;
    }
  }

  // For testing: reading TSP instance file; computing some MSTs
  public static void main(String args[]) throws Exception {
    TSPInstance tsp0 = parse(args[0]);
    System.out.println(args[0]);
    System.out.println("dim: " + tsp0.dim());
    System.out.println("distance matrix:");
    for (int i = 0; i < tsp0.dim(); i++) {
      for (int j = 0; j < tsp0.dim(); j++)
        System.out.print(tsp0.dist(i,j) + " ");
      System.out.println();
    }
    // Computing weightMST() of first 3 cities
    ArrayList<Integer> cities = new ArrayList<>();
    for (int i = 0; i < Math.min(tsp0.dim(), 3); i++)
      cities.add(i);
    System.out.println("weightMST(first_3_cities) = " + tsp0.weightMST(cities));
    // Computing weightMST() of all even cities
    cities.clear();
    for (int i = 0; i < tsp0.dim(); i += 2)
      cities.add(i);
    System.out.println("weightMST(even_cities) = " + tsp0.weightMST(cities));
    // Computing weightMST() of entire set of cities
    cities.clear();
    for (int i = 0; i < tsp0.dim(); i++)
      cities.add(i);
    System.out.println("weightMST(all_cities) = " + tsp0.weightMST(cities));
  }


  /////////////////////////////////////////////////////////////////////////////
  // Weight of a minimum spanning tree connecting a subset of cities
  // using [https://en.wikipedia.org/wiki/Kruskal's_algorithm]

  // Data class representing edges
  private class Edge {
    int source;  // source vertex
    int target;  // target vertex
    int weight;  // weight = distance between source and target
    Edge(int i, int j, int w) { source = i; target = j; weight = w; }
  }

  // Returns the weight of a minimum spanning tree connecting the given cities.
  public int weightMST(Collection<Integer> cities) {
    // Set up forest stored in parent and height arrays
    int[] parent = new int[dim];  // maps vertices to their parent
    int[] height = new int[dim];  // maps vertices to the height of their tree
    // Initialise parent (to the identity map) and height (to the zero map).
    for (int v = 0; v < dim; v++) {
      parent[v] = v;
      height[v] = 0;  // height[v] == 0 means vertex v is ignored
    }
    // Initialise height to 1 for vertices that shall not be ignored
    for (int v : cities)
      height[v] = 1;

    // Initialise the weight of the MST and the edge countdown
    int mst_weight = 0;
    int mst_countdown = cities.size() - 1;  // number of MST edges to find

    // Iterate over edges in order of increasing weight
    for (Edge e : edges) {
      // Stop when all edges have been found
      if (mst_countdown <= 0)
        break;

      // Skip iteration if current edge e = (u,v) leaves the given set cities
      int u = e.source;
      int v = e.target;
      if (height[u] == 0 || height[v] == 0)
        continue;

      // FIND: Follow u and v to their respective roots
      while (u != parent[u]) { u = parent[u]; }
      while (v != parent[v]) { v = parent[v]; }

      // If these roots are different, add the current edge e to the MST
      if (u != v) {
        mst_weight += e.weight;
        mst_countdown--;

        // UNION: join trees by linking root of the smaller one to the other
        if (height[u] < height[v]) {
          parent[u] = v;
        } else {
          parent[v] = u;
          if (height[u] == height[v])
            height[u]++;
        }
      }
    }

    // Return weight of MST
    return mst_weight;
  }
}
