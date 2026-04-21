#include "Graph.h"
#include "ANN/ANN.h"


namespace NNPNet {
	class NP {
	public:
		template <typename T>
		static T np(Graph<T>& g, T r = 2) {

			// Change r if the graph is weighted
			if (g.weighted) {
				int edgeCount = 0;
				double totLength = 0;
				for (auto& edges : g.edges) {
					edgeCount += edges.size();
					for (auto& e : edges) {
						totLength += e.weight;
					}
				}
				r *= totLength / edgeCount;
			}



			// Setup ANN tree
			ANNpointArray inPoints = annAllocPts(g.nodeCount, g.outputDim);
			for (int i = 0; i < g.nodeCount; i++) {
				for (int d = 0; d < g.outputDim; d++) {
					inPoints[i][d] = (ANNcoord)g.Y[i * g.outputDim + d];
				}
			}
			ANNkd_tree* tree = new ANNkd_tree(inPoints, g.nodeCount, g.outputDim);


			// Calculate np for all nodes
			double s = Threadpool::divideWork((function<double(int, int)>)[&g, r, tree](int begin, int end) {
				double s = 0;
				int* rnnNodes = (int*)malloc(g.nodeCount * sizeof(int));
				ANNdistArray dists = new ANNdist[g.nodeCount];
				ANNidxArray idx = new ANNidx[g.nodeCount];
				ANNpoint qPoint = annAllocPt(g.outputDim);
				for (int j = begin; j < end; j++) {
					// r-nn
					int xnn = g.rnn(j, rnnNodes, r);
					std::unordered_set<int> xnnNodes;
					for (int i = 0; i < xnn; i++) {
						xnnNodes.insert(rnnNodes[i]);
					}
					if (xnn > 10000) {
						xnn = g.rnn(j, rnnNodes, r / 2);
						xnnNodes.clear();
						for (int i = 0; i < xnn; i++) {
							xnnNodes.insert(rnnNodes[i]);
						}
					}

					// k-nn
					for (int d = 0; d < g.outputDim; d++) {
						qPoint[d] = (ANNcoord)g.Y[j * g.outputDim + d];
					}
					tree->annkSearch(qPoint, xnn + 1, idx, dists, 0.0);

					// Calculate overlap
					int xOry = xnn, xAndy = 0;
					// Skip first point, as that is the point we are querying
					for (int i = 1; i < xnn + 1; i++) {
						if (xnnNodes.count(idx[i]) != 0) xAndy++;
						else xOry++;
					}
					if (xOry > 0) s += ((double)xAndy) / ((double)xOry);
				}
				free(rnnNodes);
				delete[] dists;
				delete[] idx;
				delete[] qPoint;
				return s;
				}, g.nodeCount, (function<double(double, double)>)[](double a, double b) {return a + b; });

			// Cleanup
			delete tree;
			delete[] inPoints;

			std::cout << "Neighborhood Presevation: " << std::to_string(s / g.nodeCount) << "\n";

			return s / g.nodeCount;
		};
	};
};