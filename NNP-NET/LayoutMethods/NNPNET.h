#pragma once

#include "../Graph.h"

namespace NNPNet {
	class NNPNET {
	public:
		NNPNET() {};

		void run(Graph<float>& g, Graph<double>* GT = nullptr, int embeddingSize = 50);

		int subgraphPoints = 10000, perplexity = 40, pmdsPivots = 250;
		bool pmdsEmbedding = true, fastSubgraph = true, gpu = false;

		double theta = 0.25;

	private:
		float* createPivotEmbedding(Graph<float>& g, int pivots);

		float* createPMDSEmbedding(Graph<float>& g, int& dimensions);

		void createSubnetwork(Graph<float>& inG, Graph<float>& outG, std::vector<int>& nodes);

		void createFastSubnetwork(Graph<float>& inG, Graph<float>& outG, std::vector<int>& nodes);
		int fastSubnetworkIteration(Graph<float>& in, Graph<float>& out, std::vector<int>& inNodes, std::vector<int>& outNodes, int target, int startPos = 0);
		void getFastSubnetworkOrder(Graph<float>& g, std::vector<std::pair<int, int>>& list);

		void trainNetwork(Graph<float>& g, Graph<double>* GT, float* embedding, int embeddingDim);
	
		void trainPlusInfer(float* smallEmbedding, int smallEmbeddingSize, double* gt, float* fullEmbedding, int fullEmbeddingSize, int embeddingDim, int outputDim, float* Y);

	};
};
