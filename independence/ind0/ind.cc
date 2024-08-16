// C++ program for above approach
#include <iostream>
#include <map>
#include <vector>
#include "c_chunk.h"

using namespace std;

// Recursive Function to find the
// Maximal Independent Vertex Set
vector<int> graphSets(map<int, vector<int> > graph)
{
	// Base Case - Given Graph has no nodes
	if (graph.size() == 0) {
		return vector<int>();
	}

	// Base Case - Given Graph has 1 node
	if (graph.size() == 1) {
		vector<int> v;
		for (auto const& element : graph) {
			v.push_back(element.first);
		}
		return v;
	}

	// Select a vertex from the graph
	int vCurrent = graph.begin()->first;

	// Case 1 - Proceed removing the selected vertex
	// from the Maximal Set
	map<int, vector<int> > graph2(graph);

	// Delete current vertex from the Graph
	graph2.erase(vCurrent);

	// Recursive call - Gets Maximal Set,
	// assuming current Vertex not selected
	vector<int> res1 = graphSets(graph2);

	// Case 2 - Proceed considering the selected vertex
	// as part of the Maximal Set

	// Loop through its neighbours
	for (auto v : graph.at(vCurrent)) {
		// Delete neighbor from the current subgraph
		if (graph2.count(v)) {
			graph2.erase(v);
		}
	}

	// This result set contains vCurrent,
	// and the result of recursive call assuming neighbors
	// of vCurrent are not selected
	vector<int> res2;
	res2.push_back(vCurrent);
	vector<int> res2Sub = graphSets(graph2);
	res2.insert(res2.end(), res2Sub.begin(), res2Sub.end());

	// Our final result is the one which is bigger, return
	// it
	if (res1.size() > res2.size()) {
		return res1;
	}
	return res2;
}


// Driver Code
int main()
{
	//int V = 8;

	// Defines edges
	// int E[][2] = { { 1, 2 }, { 1, 3 }, { 2, 4 },
	// 			{ 5, 6 }, { 6, 7 }, { 4, 8 } };
 


	map<int, vector<int> > graph;

	// Constructs Graph as a dictionary of the following
	// format- graph[VertexNumber V] = list[Neighbors of
	// Vertex V]
	for (int i = 0; i < sizeof(E) / sizeof(E[0]); i++) {
		int v1 = E[i][0];
		int v2 = E[i][1];
		if (graph.count(v1) == 0) {
			graph[v1] = vector<int>();
		}
		if (graph.count(v2) == 0) {
			graph[v2] = vector<int>();
		}
		graph[v1].push_back(v2);
		graph[v2].push_back(v1);
	}

	// Recursive call considering all vertices in the
	// maximum independent set
	vector<int> maximalIndependentSet = graphSets(graph);

    
	// Prints the Result
	// for (auto i : maximalIndependentSet) {
	// 	cout << i << " ";
	// }
    bool first = true;
    for (auto edge : E){
        if(first){
            cout << "(" << maximalIndependentSet.size() << ",( ("<< edge[0] << "," << edge[1] << ")";
        }
        else
            cout << ",("<< edge[0] << "," << edge[1] << ")";
    }

    cout << ")" << ")";
	cout << endl;

	return 0;
}
