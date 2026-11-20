#include "NNPNET.h"

#include "PivotMDS.h"
#include "tsNET.h"
#include "../SubgraphMethods.h"
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
		TIME(embedding = createPMDSEmbedding(g, pivots, nullptr, nullptr), "Create Embedding");
	}
	else {
		TIME(embedding = createPivotEmbedding(g, pivots), "Create Embedding");
	}

	trainNetwork(g, GT, embedding, pivots);
	free(embedding);
}

void NNPNet::NNPNET::run(Graph<float>& g, int* pivotPoints, Graph<double>* GT, int pivots)
{
	float* embedding;
	if (pmdsEmbedding) {
		TIME(embedding = createPMDSEmbedding(g, pivots, nullptr, &pivotPoints), "Create Embedding");
	}
	else {
		TIME(embedding = createPivotEmbedding(g, pivots), "Create Embedding");
	}

	//trainNetwork(g, GT, embedding, pivots);
	infer(embedding, g.nodeCount, g.outputDim, g.Y);
	free(embedding);
}

int* NNPNet::NNPNET::run(Graph<float>& g, bool* inAll, Graph<double>* GT, int pivots)
{
	float* embedding;
	int* pivotPoints;
	if (pmdsEmbedding) {
		TIME(embedding = createPMDSEmbedding(g, pivots, inAll, &pivotPoints), "Create Embedding");
	}
	else {
		TIME(embedding = createPivotEmbedding(g, pivots), "Create Embedding");
	}

	trainNetwork(g, GT, embedding, pivots);
	free(embedding);
	return pivotPoints;
}

float* NNPNet::NNPNET::createPivotEmbedding(Graph<float>& g, int pivots)
{
	float* embedding = (float*)malloc(((size_t)g.nodeCount) * ((size_t)pivots) * sizeof(float));
	size_t n = g.nodeCount;
	size_t pivot = 0;
	float* lowest = (float*)malloc(sizeof(float) * n);
	for (size_t i = 0; i < pivots; i++) {
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
			size_t p_i = i * n;
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
	for (size_t i = 0; i < g.nodeCount; i++) {
		for (size_t j = 0; j < pivots; j++) {
			if (embedding[j * ((size_t)g.nodeCount) + i] > biggest) {
				biggest = embedding[j * ((size_t)g.nodeCount) + i];
			}
		}
	}
	for (size_t i = 0; i < g.nodeCount; i++) {
		for (size_t j = 0; j < pivots; j++) {
			embedding[j * ((size_t)g.nodeCount) + i] /= biggest;
		}
	}
	
	return embedding;
}

float* NNPNet::NNPNET::createPMDSEmbedding(Graph<float>& g, int& dimensions, bool* inAll, int** pivotPoints)
{
	float* embedding = (float*)malloc(((size_t)g.nodeCount) * ((size_t)dimensions) * sizeof(float));
	size_t n = g.nodeCount;
	float* backupY = g.Y;
	size_t backupDim = g.outputDim;
	g.Y = embedding;
	g.outputDim = dimensions;

	if (useFloats) {
		
		pmds_f.setNumberOfPivots(pmdsPivots);
		if (inAll != nullptr) {
			(*pivotPoints) = pmds_f.call(g, inAll);
		}
		else if (pivotPoints != nullptr){
			pmds_f.call(g, *pivotPoints);
		}
		else {
			pmds_f.call(g);
		}
	}
	else {
		Graph<double> g_d(g);
		
		pmds_d.setNumberOfPivots(pmdsPivots);
		if (inAll != nullptr) {
			(*pivotPoints) = pmds_d.call(g_d, inAll);
		}
		else if (pivotPoints != nullptr) {
			pmds_d.call(g_d, *pivotPoints);
		}
		else {
			pmds_d.call(g_d);
		}
		for (int i = 0; i < ((size_t)g.nodeCount) * ((size_t)dimensions); i++) {
			g.Y[i] = (float)g_d.Y[i];
		}
	}

	bool hasNaN = false;
	for (size_t i = 0; i < ((size_t)g.nodeCount) * ((size_t)dimensions); i++) {
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

void NNPNet::NNPNET::trainNetwork(Graph<float>& g, Graph<double>* GT, float* embedding, int embeddingDim)
{
	Graph<float> subG(g.outputDim);
	std::vector<int> nodes;

	if (GT != nullptr) {
		// Convert to output labels
		// First train on a subset for faster convergence
		if (g.nodeCount > subgraphPoints * 1.5) {
			TIME(SubgraphMethods::ASSC(g, subG, nodes, subgraphPoints), "Create fast subgraph");

			float* smallEmbedding = (float*)malloc(subG.nodeCount * embeddingDim * sizeof(float));
			for (int i = 0; i < subG.nodeCount; i++) {
				for (int d = 0; d < embeddingDim; d++) {
					smallEmbedding[i*embeddingDim + d] = embedding[((size_t)nodes[i]) * ((size_t)embeddingDim) + (size_t)d];
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
		TIME(SubgraphMethods::ASSC(g, subG, nodes, subgraphPoints), "Create fast subgraph");
		multivariteSubgraph(g, subG, nodes);

		TSNET<double> tsnet;
		tsnet.perp = perplexity;
		tsnet.featureWeight = featureWeight;
		Graph<double> subD(subG);
		TIME(tsnet.tsNETStar(subD, theta), "creating ground truth");

		subD.normalize();
		//subD.saveToVNA("GT" + std::to_string(featureWeight) + ".vna");
	
		// Convert to output labels
		float* smallEmbedding = (float*)malloc(subG.nodeCount * embeddingDim * sizeof(float));
		for (int i = 0; i < subG.nodeCount; i++) {
			for (int d = 0; d < embeddingDim; d++) {
				smallEmbedding[i * embeddingDim + d] = embedding[((size_t)nodes[i]) * ((size_t)embeddingDim) + (size_t)d];
			}
		}

		if (featureWeight > 0 && g.featureDimensions > 0) {
			float* _fullEmbedding, * _smallEmbedding;
			addMultivariateInformation(embedding, &_fullEmbedding, smallEmbedding, &_smallEmbedding, g, nodes, embeddingDim);
			TIME(trainPlusInfer(_smallEmbedding, subG.nodeCount, subD.Y, _fullEmbedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
			free(_fullEmbedding);
			free(_smallEmbedding);
		}
		else {
			TIME(trainPlusInfer(smallEmbedding, subG.nodeCount, subD.Y, embedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
		}
		free(smallEmbedding);
	}
	else {
		TSNET<double> tsnet;
		tsnet.perp = perplexity;
		tsnet.featureWeight = featureWeight;
		Graph<double> gd(g);
		TIME(tsnet.tsNETStar(gd, theta), "creating ground truth");
		gd.normalize();
		//gd.saveToVNA("GT" + std::to_string(featureWeight) + ".vna");

		// Convert to output labels
		if (featureWeight > 0 && g.featureDimensions > 0) {
			float* _fullEmbedding;
			addMultivariateInformation(embedding, &_fullEmbedding, nullptr, nullptr, g, nodes, embeddingDim);
			TIME(trainPlusInfer(_fullEmbedding, g.nodeCount, gd.Y, _fullEmbedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
			free(_fullEmbedding);
		}
		else {
			TIME(trainPlusInfer(embedding, g.nodeCount, gd.Y, embedding, g.nodeCount, embeddingDim, g.outputDim, g.Y), "Train and Inference");
		}
	}
}

void NNPNet::NNPNET::multivariteSubgraph(Graph<float>& g, Graph<float>& subG, std::vector<int>& nodes)
{
	if (featureWeight <= 0 || g.featureDimensions <= 0) return;

	subG.features = (float*)malloc(g.featureDimensions * subG.nodeCount * sizeof(float));
	subG.featureDimensions = g.featureDimensions;
	for (int i = 0; i < subG.nodeCount; i++) {
		memcpy(subG.features + (i * g.featureDimensions), g.features + (nodes[i] * g.featureDimensions), sizeof(float) * g.featureDimensions);
	}
}

void NNPNet::NNPNET::addMultivariateInformation(float* fullEmbeddingIn, float** fullEmbeddingOut, float* smallEmbeddingIn, float** smallEmbeddingOut, Graph<float>& g, std::vector<int>& nodesSubgraph, int& embeddingDim)
{
	if (g.featureDimensions == 0) {
		(*fullEmbeddingOut) = fullEmbeddingIn;
		(*smallEmbeddingOut) = smallEmbeddingIn;
		return;
	}

	int newEmbeddingSize = embeddingDim + g.featureDimensions;
	(*fullEmbeddingOut) = (float*)malloc(sizeof(float) * newEmbeddingSize * g.nodeCount);

	for (int i = 0; i < g.nodeCount; i++) {
		memcpy(*fullEmbeddingOut + (i * newEmbeddingSize), fullEmbeddingIn + (i * embeddingDim), sizeof(float) * embeddingDim);
		memcpy(*fullEmbeddingOut + (i * newEmbeddingSize + embeddingDim), g.features + (i * g.featureDimensions), sizeof(float) * g.featureDimensions);
	}
	if (smallEmbeddingIn != nullptr) {
		(*smallEmbeddingOut) = (float*)malloc(sizeof(float) * newEmbeddingSize * nodesSubgraph.size());
		for (int i = 0; i < nodesSubgraph.size(); i++) {
			memcpy(*smallEmbeddingOut + (i * newEmbeddingSize), smallEmbeddingIn + (i * embeddingDim), sizeof(float) * embeddingDim);
			memcpy(*smallEmbeddingOut + (i * newEmbeddingSize + embeddingDim), g.features + (nodesSubgraph[i] * g.featureDimensions), sizeof(float) * g.featureDimensions);
		}
	}

	embeddingDim = newEmbeddingSize;
}

static float* _gt = nullptr;
static float* _fullEmbedding = nullptr;
static float* _smallEmbedding = nullptr;
static bool first = true;
static void* loc = nullptr;
static int _embeddingDim;

// PYBIND11_EMBEDDED_MODULE(getLists, m) {
// 	m.def("getSmallEmbedding", [](size_t smallEmbeddingSize, size_t embeddingDim) {
// 		return py::array_t<float>({ smallEmbeddingSize, embeddingDim }, _smallEmbedding);
// 		});
// 	m.def("getGt", [](size_t smallEmbeddingSize, size_t outputDim) {
// 		return py::array_t<float>({ smallEmbeddingSize, outputDim }, _gt);
// 		});
// 	m.def("getFullEmbedding", [](size_t fullEmbeddingSize, size_t embeddingDim) {
// 		return py::array_t<float>({ fullEmbeddingSize, embeddingDim }, _fullEmbedding);
// 		});
// }

void NNPNet::NNPNET::trainPlusInfer(float* smallEmbedding, int smallEmbeddingSize, double* gt, float* fullEmbedding, int fullEmbeddingSize, int embeddingDim, int outputDim, float* Y)
{
	float* fgt = (float*)malloc(smallEmbeddingSize * outputDim * sizeof(float));
	for (int i = 0; i < smallEmbeddingSize * outputDim; i++) {
		fgt[i] = (float)gt[i];
	}
	_gt = fgt;
	_fullEmbedding = fullEmbedding;
	_smallEmbedding = smallEmbedding;
	_embeddingDim = embeddingDim;
	// Else the loss function gives an error
	smallEmbeddingSize -= (smallEmbeddingSize%64);

	// Call the python script
	if (loc != nullptr) delete (py::dict*)loc;
	auto l = new py::dict(
		"smallEmbeddingSize"_a = smallEmbeddingSize,
		"fullEmbeddingSize"_a = fullEmbeddingSize,
		"embeddingDim"_a = embeddingDim,
		"trainEpochs"_a = trainingEpochs,
		"batchSize"_a = batchSize,
		"outputDim"_a = outputDim,
		"smallEmbedding"_a = py::array_t<float>({ smallEmbeddingSize, embeddingDim }, _smallEmbedding),
		"gt"_a = py::array_t<float>({ smallEmbeddingSize, outputDim }, _gt),
		"fullEmbedding"_a = py::array_t<float>({ fullEmbeddingSize, embeddingDim }, _fullEmbedding)
		);
	loc = l;
	

	py::exec(R"(
import keras
from keras import layers
import tensorflow as tf
)"
 + (gpu ? std::string() : std::string("tf.config.set_visible_devices([], 'GPU')\n")) + 
R"(import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[embeddingDim], batch_size=batchSize),
    tf.keras.layers.Dense(256, activation="leaky_relu"),
    tf.keras.layers.Dense(512, activation="leaky_relu"),
    tf.keras.layers.Dense(256, activation="leaky_relu"),
    tf.keras.layers.Dense(outputDim)
    ])
model.compile(optimizer='Adam',
              loss=tf.keras.losses.MeanSquaredError())
    
#callback = keras.callbacks.EarlyStopping(monitor='loss',patience=2, min_delta=0.0001)

model.fit(smallEmbedding, gt, epochs=trainEpochs, batch_size=batchSize) #callbacks=[callback]
    
outPredictions = model.predict(fullEmbedding, batch_size=4096)
)", py::globals(), (*(py::dict*)loc));

	float* out = (float*)(*(py::dict*)loc)["outPredictions"].cast<py::array>().data();

	memcpy(Y, out, fullEmbeddingSize * outputDim * sizeof(float));

	free(fgt);
	
}

void NNPNet::NNPNET::infer(float* fullEmbedding, int fullEmbeddingSize, int outputDim, float* Y)
{
	_fullEmbedding = fullEmbedding;
	(*(py::dict*)loc)["fullEmbeddingSize"] = fullEmbeddingSize;
	(*(py::dict*)loc)["fullEmbedding"] = py::array_t<float>({ fullEmbeddingSize, _embeddingDim }, _fullEmbedding);
	py::exec(R"(
outPredictions = model.predict(fullEmbedding, batch_size=4096)
)", py::globals(), (*(py::dict*)loc));
	float* out = (float*)(*(py::dict*)loc)["outPredictions"].cast<py::array>().data();

	memcpy(Y, out, fullEmbeddingSize * outputDim * sizeof(float));
}
