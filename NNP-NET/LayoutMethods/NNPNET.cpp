#include "NNPNET.h"

#include "PivotMDS.h"
#include "tsNET.h"
#include "../Utils.h"
#include "../Threading.h"

#include <math.h>

#include "pybind11/embed.h"
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

#define ALLOW_LESS_STRICT_MERGING
#define SET_ORDERING
//#define REVERSE_ORDER

void NNPNet::NNPNET::run(Graph<float>& g, Graph<double>* GT, int pivots)
{
	float* embedding;
	if (pmdsEmbedding) {
		TIME(embedding = createPMDSEmbedding(g, pivots), "Create Embedding");
	}
	else {
		TIME(embedding = createPivotEmbedding(g, pivots), "Create Embedding");
	}

	trainNetwork(g, GT, embedding, pivots);
	free(embedding);
}

float* NNPNet::NNPNET::createPivotEmbedding(Graph<float>& g, int pivots)
{
	float* embedding = (float*)malloc(g.nodeCount * pivots * sizeof(float));
	int n = g.nodeCount;
	int pivot = 0;
	float* lowest = (float*)malloc(sizeof(float) * n);
	for (int i = 0; i < pivots; i++) {
		g.getDistances(pivot, embedding + i*n);

		if (i + 1 < pivots) {
			float highest = 0;
			if (i == 0) {
				std::memcpy(lowest, embedding, n * sizeof(float));
				for (int j = 0; j < n; j++) {
					if (lowest[j] > highest) {
						highest = lowest[j];
						pivot = j;
					}
				}
				continue;
			}
			int p_i = i * n;
			for (int j = 0; j < n; j++) {
				if (embedding[p_i] < lowest[j]) {
					lowest[j] = embedding[p_i];
				}
				if (lowest[j] > highest) {
					highest = lowest[j];
					pivot = j;
				}
				p_i++;
			}
		}
	}
	float biggest = 0;
	for (int i = 0; i < g.nodeCount; i++) {
		for (int j = 0; j < pivots; j++) {
			if (embedding[j * g.nodeCount + i] > biggest) {
				biggest = embedding[j * g.nodeCount + i];
			}
		}
	}
	for (int i = 0; i < g.nodeCount; i++) {
		for (int j = 0; j < pivots; j++) {
			embedding[j * g.nodeCount + i] /= biggest;
		}
	}
	
	return embedding;
}

float* NNPNet::NNPNET::createPMDSEmbedding(Graph<float>& g, int& dimensions)
{
	float* embedding = (float*)malloc(g.nodeCount * dimensions * sizeof(float));
	int n = g.nodeCount;
	float* backupY = g.Y;
	int backupDim = g.outputDim;
	g.Y = embedding;
	g.outputDim = dimensions;

	PivotMDS<float> pmds;
	pmds.setNumberOfPivots(pmdsPivots);
	pmds.call(g);

	bool hasNaN = false;
	for (int i = 0; i < g.nodeCount * dimensions; i++) {
		if (std::isnan(embedding[i])) {
			hasNaN = true;
			break;
		}
	}

	if (!hasNaN)g.normalize();
	g.Y = backupY;
	g.outputDim = backupDim;

	if (hasNaN) {
		std::cout << "PMDS embedding contained NaN's, using pivots as the embedding\n";
		free(embedding);
		return createPivotEmbedding(g, dimensions);
	}
	return embedding;
}

void NNPNet::NNPNET::createSubnetwork(Graph<float>& g, Graph<float>& outG, std::vector<int>& nodes)
{
	int targetPoints = subgraphPoints;
	outG.Y = (float*)malloc(targetPoints * g.outputDim * sizeof(float));
	float* embedding = (float*)malloc(g.nodeCount * sizeof(float));
	int n = g.nodeCount;
	int pivot = 0;
	float* lowest = (float*)malloc(sizeof(float) * n);
	outG.edges.resize(targetPoints);
	for (int i = 0; i < targetPoints; i++) {
		g.getDistances(pivot, embedding);
		nodes.push_back(pivot);

		// Next pivot point
		if (i + 1 < targetPoints) {
			float highest = 0;
			if (i == 0) {
				std::memcpy(lowest, embedding, n * sizeof(float));
				for (int j = 0; j < n; j++) {
					if (lowest[j] > highest) {
						highest = lowest[j];
						pivot = j;
					}
				}
				continue;
			}
			for (int j = 0; j < n; j++) {
				if (embedding[j] < lowest[j]) {
					lowest[j] = embedding[j];
				}
				if (lowest[j] > highest) {
					highest = lowest[j];
					pivot = j;
				}
			}
		}

		// Add edges
		float l = 9999999999;
		for (int j = 0; j < i; j++) {
			if (embedding[nodes[j]] < l) {
				l = embedding[nodes[j]];
			}
		}

		for (int j = 0; j < i; j++) {
			outG.edges[i].push_back(Edge(j, embedding[nodes[j]]));
			outG.edges[j].push_back(Edge(i, embedding[nodes[j]]));
		}
	}
	free(lowest);
	free(embedding);
	outG.nodeCount = targetPoints;

}

void NNPNet::NNPNET::createFastSubnetwork(Graph<float>& inG, Graph<float>& outG, std::vector<int>& nodes)
{
	int target = subgraphPoints;
	Graph<float> temp(inG.outputDim);
	Graph<float> temp2(inG.outputDim);
	outG.weighted = true;
	temp.weighted = true;
	temp2.weighted = true;
	
	std::vector<int> tempNodes;
	std::vector<int> tempNodes2;
	// fill temp nodes with all numbers
	tempNodes.resize(inG.nodeCount);
	for (int i = 0; i < inG.nodeCount; i++) {
		tempNodes[i] = i;
	}
	int currentNodeCount = inG.nodeCount;
	int startPos = 0;
	startPos = fastSubnetworkIteration(inG, temp2, tempNodes, tempNodes2, target, startPos);
	currentNodeCount = temp2.nodeCount;
	bool currentIsFirst = false;
	float reduction = 0;
	while (currentNodeCount > target && reduction < 0.95) {
		std::cout << currentNodeCount << "\n";
		if (currentIsFirst) {
			startPos = fastSubnetworkIteration(temp, temp2, tempNodes, tempNodes2, target, startPos);
			currentNodeCount = temp2.nodeCount;
			currentIsFirst = false;
			reduction = ((float)temp2.nodeCount) / ((float)temp.nodeCount);
		}
		else {
			startPos = fastSubnetworkIteration(temp2, temp, tempNodes2, tempNodes, target, startPos);
			currentNodeCount = temp.nodeCount;
			currentIsFirst = true;
			reduction = ((float)temp.nodeCount) / ((float)temp2.nodeCount);
		}
	}
	if (currentIsFirst) {
		if (currentNodeCount > target) {
			std::cout << "Does not collapse nicelly, reverting to pivot method to reduced graph\n";
			temp.weighted = true;
			createSubnetwork(temp, outG, nodes);
		}
		else {
			outG = temp;
			for (int i : tempNodes) {
				nodes.push_back(i);
			}
		}
	}
	else {
		if (currentNodeCount > target) {
			std::cout << "Does not collapse nicelly, reverting to pivot method to reduced graph\n";
			temp2.weighted = true;
			createSubnetwork(temp2, outG, nodes);
		}
		else {
			outG = temp2;
			for (int i : tempNodes2) {
				nodes.push_back(i);
			}
		}
	}
	outG.Y = (float*)malloc(sizeof(float) * outG.nodeCount * outG.outputDim);
	std::cout << "Number of subgraph points reached: " << outG.nodeCount << "\n";
}

void NNPNet::NNPNET::getFastSubnetworkOrder(Graph<float>& g, std::vector<std::pair<int, int>>& list)
{
	// Fill list
	list.resize(g.nodeCount);
	for (int i = 0; i < g.nodeCount; i++) {
		list[i].first = g.edges[i].size();
		list[i].second = i;
	}
	std::vector<std::pair<int, int>> list2;
	list2.resize(g.nodeCount);
	// Radix sort
	int counts[256];
	for (int i = 0; i < 2; i++) {
		// Reset counts
		for (int j = 0; j < 256; j++) counts[j] = 0;
		// Count
		int off = i * 2;
		for (int j = 0; j < g.nodeCount; j++) {
			counts[*((unsigned char*)(list.data() + j) + off)]++;
		}
		// Set offsets
		int curr = 0;
#ifdef REVERSE_ORDER
		for (int j = 255; j >= 0; j--) {
#else
		for (int j = 0; j < 256; j++) {
#endif
			int next = curr + counts[j];
			counts[j] = curr;
			curr = next;
		}
		// Move into new list
		for (int j = 0; j < g.nodeCount; j++) {
			list2[counts[*((unsigned char*)(list.data() + j) + off)]++] = list[j];
		}

		// Again, but from list2 -> list
		// Reset counts
		for (int j = 0; j < 256; j++) counts[j] = 0;
		// Count
		off = i * 2 + 1;
		for (int j = 0; j < g.nodeCount; j++) {
			counts[*((unsigned char*)(list2.data() + j) + off)]++;
		}
		// Set offsets
		curr = 0;
#ifdef REVERSE_ORDER
		for (int j = 255; j >= 0; j--) {
#else
		for (int j = 0; j < 256; j++) {
#endif
			int next = curr + counts[j];
			counts[j] = curr;
			curr = next;
		}
		// Move into new list
		for (int j = 0; j < g.nodeCount; j++) {
			list[counts[*((unsigned char*)(list2.data() + j) + off)]++] = list2[j];
		}
	}
}

int NNPNet::NNPNET::fastSubnetworkIteration(Graph<float>& in, Graph<float>& out, std::vector<int>& inNodes, std::vector<int>& outNodes, int target, int startPos)
{
	out.edges.clear();
	out.nodeCount = 0;
	outNodes.clear();
	int* partOfNode = (int*)malloc(sizeof(int) * in.nodeCount);
	double* distanceFromCenter = (double*)malloc(sizeof(double) * in.nodeCount);
	startPos -= 1;
	if (startPos == -1) {
		startPos += in.nodeCount;
	}
	// Set all nodes as not part of a new node
	for (int i = 0; i < in.nodeCount; i++) {
		partOfNode[i] = -1;
		distanceFromCenter[i] = 0;
	}
	// Create clusters
	std::cout << in.nodeCount << " -> ";
	int currentNodeCount = in.nodeCount;
#ifdef SET_ORDERING
	std::vector<std::pair<int, int>> order;
	getFastSubnetworkOrder(in, order);
	for (int j = 0; j < in.nodeCount; j++) {
#ifdef REVERSE_ORDER
		int left = currentNodeCount - target + 10;
		while (order[j].first > left && j < in.nodeCount - 1) {
			j++;
		}
#endif
		int i = order[j].second;
#else
	for (int i = (startPos + 1)%in.nodeCount; i != startPos; i = (i + 1)%in.nodeCount) {
#endif
		if (partOfNode[i] == -1) {
#ifndef ALLOW_LESS_STRICT_MERGING
			bool neighborsFree = true;
			// Only if none of the nodes that would be included
			// are already part of another cluster
			for (Edge& e : in.edges[i]) {
				if (partOfNode[e.other] != -1) {
					neighborsFree = false;
					break;
				}
			}
#endif

			// Add it to the list of nodes
#ifdef ALLOW_LESS_STRICT_MERGING
			{
#else
			if (neighborsFree) {
#endif
				partOfNode[i] = out.nodeCount;
				int count = 0;
				for (Edge<float>& e : in.edges[i]) {
					if (partOfNode[e.other] == -1 && e.other != i) {
						partOfNode[e.other] = out.nodeCount;
						distanceFromCenter[e.other] = e.weight;
						count++;
					}
				}
				outNodes.push_back(inNodes[i]);
				currentNodeCount -= count;
				out.nodeCount++;
				if (currentNodeCount <= target) {
					break;
				}

			}
		}
	}
	// Add everything that has not an assigned node yet
	for (int i = 0; i < in.nodeCount; i++) {
		if (partOfNode[i] == -1) {
			outNodes.push_back(inNodes[i]);
			partOfNode[i] = out.nodeCount;
			out.nodeCount++;
		}
	}
	int nextStart = (startPos + out.nodeCount / 4) % out.nodeCount;
	// Edges
	std::vector<std::unordered_map<int, double>> edges;
	edges.resize(out.nodeCount);
	for (int i = 0; i < in.nodeCount; i++) {
		for (Edge e : in.edges[i]) {
			if (partOfNode[e.other] == partOfNode[i]) continue;
			double dist = distanceFromCenter[i] + distanceFromCenter[e.other] + e.weight;
			if (edges[partOfNode[i]].count(partOfNode[e.other]) == 0 || edges[partOfNode[i]][partOfNode[e.other]] > dist) {
				edges[partOfNode[i]][partOfNode[e.other]] = dist;
			}
		}
	}
	// Copy edges into the right format
	out.edges.resize(out.nodeCount);
	for (int i = 0; i < out.nodeCount; i++) {
		for (auto& edge : edges[i]) {
			out.edges[i].push_back(Edge<float>(edge.first, edge.second));
		}
	}

	free(partOfNode);
	return nextStart;
}

void NNPNet::NNPNET::trainNetwork(Graph<float>& g, Graph<double>* GT, float* embedding, int embeddingDim)
{
	Graph<float> subG(g.outputDim);
	std::vector<int> nodes;

	if (GT != nullptr) {
		// Convert to output labels
		// First train on a subset for faster convergence
		if (g.nodeCount > subgraphPoints * 1.5) {
			if (fastSubgraph) {
				TIME(createFastSubnetwork(g, subG, nodes), "Create subgraph");
			}
			else {
				TIME(createSubnetwork(g, subG, nodes), "Create subgraph");
			}

			float* smallEmbedding = (float*)malloc(subG.nodeCount * embeddingDim * sizeof(float));
			for (int i = 0; i < subG.nodeCount; i++) {
				for (int d = 0; d < embeddingDim; d++) {
					smallEmbedding[i*embeddingDim + d] = embedding[nodes[i] * embeddingDim + d];
				}
			}
			double* outputLabels = (double*)malloc(subG.nodeCount * GT->outputDim * sizeof(double));
			for (int i = 0; i < subG.nodeCount; i++) {
				for (int d = 0; d < GT->outputDim; d++) {
					outputLabels[i * GT->outputDim + d] = GT->Y[nodes[i] * GT->outputDim + d];
				}
			}

			TIME(trainPlusInfer(smallEmbedding, subG.nodeCount, outputLabels, embedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
			free(smallEmbedding);
			free(outputLabels);
			return;
		}
		TIME(trainPlusInfer(embedding, g.nodeCount, GT->Y, embedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
		
	}
	else if (g.nodeCount > subgraphPoints*1.5) {
		if (fastSubgraph) {
			TIME(createFastSubnetwork(g, subG, nodes), "Create fast subgraph");
		}
		else {
			TIME(createSubnetwork(g, subG, nodes), "Create subgraph");
		}
		TSNET<double> tsnet;
		tsnet.perp = perplexity;
		Graph<double> subD(subG);
		TIME(tsnet.tsNETStar(subD, theta), "creating ground truth");

		subD.normalize();
	
		// Convert to output labels
		float* smallEmbedding = (float*)malloc(subG.nodeCount * embeddingDim * sizeof(float));
		for (int i = 0; i < subG.nodeCount; i++) {
			for (int d = 0; d < embeddingDim; d++) {
				smallEmbedding[i * embeddingDim + d] = embedding[nodes[i] * embeddingDim + d];
			}
		}

		TIME(trainPlusInfer(smallEmbedding, subG.nodeCount, subD.Y, embedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
		free(smallEmbedding);
	}
	else {
		TSNET<double> tsnet;
		tsnet.perp = perplexity;
		Graph<double> gd(g);
		TIME(tsnet.tsNETStar(gd, theta), "creating ground truth");
		gd.normalize();

		// Convert to output labels
		TIME(trainPlusInfer(embedding, g.nodeCount, gd.Y, embedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
	}
}

static float* _gt = nullptr;
static float* _fullEmbedding = nullptr;
static float* _smallEmbedding = nullptr;
static bool first = true;

PYBIND11_EMBEDDED_MODULE(getLists, m) {
	m.def("getSmallEmbedding", [](int smallEmbeddingSize, int embeddingDim) {
		return py::array_t<float>({ smallEmbeddingSize, embeddingDim }, _smallEmbedding);
		});
	m.def("getGt", [](int smallEmbeddingSize, int outputDim) {
		return py::array_t<float>({ smallEmbeddingSize, outputDim }, _gt);
		});
	m.def("getFullEmbedding", [](int fullEmbeddingSize, int embeddingDim) {
		return py::array_t<float>({ fullEmbeddingSize, embeddingDim }, _fullEmbedding);
		});
}

void NNPNet::NNPNET::trainPlusInfer(float* smallEmbedding, int smallEmbeddingSize, double* gt, float* fullEmbedding, int fullEmbeddingSize, int embeddingDim, int outputDim, float* Y)
{
	float* fgt = (float*)malloc(smallEmbeddingSize * outputDim * sizeof(float));
	for (int i = 0; i < smallEmbeddingSize * outputDim; i++) {
		fgt[i] = (float)gt[i];
	}
	_gt = fgt;
	_fullEmbedding = fullEmbedding;
	_smallEmbedding = smallEmbedding;
	// Else the loss function gives an error
	smallEmbeddingSize -= (smallEmbeddingSize%64);

	// Call the python script
	
	auto loc = py::dict(
		"smallEmbeddingSize"_a = smallEmbeddingSize,
		"fullEmbeddingSize"_a = fullEmbeddingSize,
		"embeddingDim"_a = embeddingDim,
		"trainEpochs"_a = trainingEpochs,
		"batchSize"_a = batchSize,
		"outputDim"_a = outputDim
		);

	py::exec(R"(
import keras
from keras import layers
import tensorflow as tf
)"
 + (gpu ? std::string() : std::string("tf.config.set_visible_devices([], 'GPU')\n")) + 
R"(import numpy as np
print("Imported standard")

import getLists

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[embeddingDim], batch_size=batchSize),
    tf.keras.layers.Dense(256, activation="leaky_relu"),
    tf.keras.layers.Dense(512, activation="leaky_relu"),
    tf.keras.layers.Dense(256, activation="leaky_relu"),
    tf.keras.layers.Dense(outputDim)
    ])
model.compile(optimizer='Adam',
              loss=tf.keras.losses.MeanSquaredError())
    
model.fit(getLists.getSmallEmbedding(smallEmbeddingSize, embeddingDim), getLists.getGt(smallEmbeddingSize, outputDim), epochs=trainEpochs, batch_size=batchSize)
    
outPredictions = model.predict(getLists.getFullEmbedding(fullEmbeddingSize, embeddingDim), batch_size=4096)
)", py::globals(), loc);

	float* out = (float*)loc["outPredictions"].cast<py::array>().data();

	memcpy(Y, out, fullEmbeddingSize * outputDim * sizeof(float));

	free(fgt);
	
}
