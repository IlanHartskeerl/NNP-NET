#include "../Graph.h"
#include "PivotMDS.h"
#include "tsNET.h"
#include "../Smoothing.h"
#include <mutex>
#include "../SubgraphMethods.h"

//#define FL_VECTORIZATION

namespace NNPNet {

	template<class T>
	class GLNet {
	public:
		GLNet() {};

		void run(Graph<T>& g) {
			Graph<T> subgraph(g.outputDim);
			std::vector<int> subNodes;
			if (subpoints > g.nodeCount / reductionAmount) subpoints = g.nodeCount / reductionAmount;
			std::cout << "Subpoints: " << subpoints << "\n";
			
			// Create subgraph
			TIMELOG(SubgraphMethods::ASSC(g, subgraph, subNodes, subpoints), "Create Subgraph");
				
			// Create Ground truth
			TSNET<double> tsnet;
			tsnet.perp = perplexity;
			Graph<double> subgraphd(subgraph);
				
			TIMELOG(tsnet.tsNETStar(subgraphd, theta), "tsNET base Layout");

			for (int i = 0; i < subgraph.nodeCount*subgraph.outputDim; i++) {
				subgraph.Y[i] = (T)subgraphd.Y[i];
			}

			// Infer all other points
			TIMELOG(inferFlood(g, subgraph, subNodes, k), "Infer all points (KNN)");

			// Smoothing
			if (smoothIter > 0) {
				SmoothingFunctions::Laplacian(g, smoothIter);
			}
			
			g.normalize();
		};

		int subpoints = 10000, reductionAmount = 10, k = 5, smoothIter, perplexity = 40;
		float theta = 0.25f;

	private:

		void inferFlood(Graph<T>& g, Graph<T>& subG, std::vector<int>& nodes, int k) {
			//std::vector<std::unordered_map<int, T>> knn;
			//TIME(g.floodKNNUnweighted(nodes, k, knn), "Flood fill KNN");
			std::vector<CustomDict<T>> knn;
			TIME(g.floodKNNUnweightedRev(nodes, k, knn), "Flood fill KNN");

			// Create pivot id translation map
			std::unordered_map<int, int> pivots;
			for (int i = 0; i < nodes.size(); i++) {
				pivots[nodes[i]] = i;
			}

			Threadpool::divideWork([this, &g, &subG, &knn, &pivots](int begin, int end) {
				int total = 0;
				for (int i = begin; i < end; i++) {
					T* Y = g.Y + i * g.outputDim;
					if (pivots.count(i) > 0) {
						for (int d = 0; d < g.outputDim; d++) {
							Y[d] = subG.Y[pivots[i] * g.outputDim + d];
						}
						continue;
					}
					T totD = 0;
					for (std::pair<int, T> nn : knn[i]) {
						totD += 1/nn.second;
					}
					// Update location
					for (int d = 0; d < g.outputDim; d++) {
						Y[d] = 0;
					}
					int j = 0;
					for (std::pair<int, T> nn : knn[i]) {
						T mult = (1/nn.second) / totD;
						for (int d = 0; d < g.outputDim; d++) {
							Y[d] += subG.Y[pivots[nn.first] * g.outputDim + d] * mult;
						}
						j++;
					}

					knn[i].clear();
				}
				}, g.nodeCount);

		}
	
	};


};