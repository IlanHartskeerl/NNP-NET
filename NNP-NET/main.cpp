// MasterThesis.cpp : Defines the entry point for the application.
//

#include <string>
#include <iostream>
#include <chrono>

#include <filesystem>
namespace fs = std::filesystem;

#include "main.h"
#include "Graph.h"

#include "LayoutMethods/NNPNET.h"
#include "LayoutMethods/PivotMDS.h"
#include "LayoutMethods/tsNET.h"

#include "Utils.h"
#include "Smoothing.h"
#include "Threading.h"

using namespace NNPNet;

#include "pybind11/embed.h"
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;


enum Argument {
	ARG_SMOOTH,
	ARG_THETA,
	ARG_GPU,
	ARG_PERP,
	ARG_METHOD,
	ARG_EMBEDDING_SIZE,
	ARG_SUBGRAPH_SIZE,
	ARG_PMDS_PIVOTS,
	ARG_OUTPATH,
	ARG_DIMENSIONS,
	ARG_TRAINING_EPOCHS,
	ARG_BATCH_SIZE,
	ARG_USE_FLOAT,

	ARG_none
};

enum Method {
	METHOD_NNPNET,
	METHOD_TSNET,
	METHOD_TSNETSTAR,
	METHOD_PMDS,
};

std::unordered_map<std::string, Argument> argMap = {
	{"-l", ARG_SMOOTH},
	{"--smoothing", ARG_SMOOTH},
	{"-t", ARG_THETA},
	{"--theta", ARG_THETA},
	{"-g", ARG_GPU},
	{"--gpu", ARG_GPU},
	{"-p", ARG_PERP},
	{"--perplexity", ARG_PERP},
	{"-m", ARG_METHOD},
	{"--method", ARG_METHOD},
	{"-e", ARG_EMBEDDING_SIZE},
	{"--embedding_size", ARG_EMBEDDING_SIZE},
	{"-s", ARG_SUBGRAPH_SIZE},
	{"--subgraph_size", ARG_SUBGRAPH_SIZE},
	{"--pmds_pivots", ARG_PMDS_PIVOTS},
	{"-o", ARG_OUTPATH},
	{"--output", ARG_OUTPATH},
	{"-d", ARG_DIMENSIONS},
	{"--dimensions", ARG_DIMENSIONS},
	{"-b", ARG_BATCH_SIZE},
	{"--batch_size", ARG_BATCH_SIZE},
	{"--training_epochs", ARG_TRAINING_EPOCHS},
	{"--use_float", ARG_USE_FLOAT},
	{"-f", ARG_USE_FLOAT}
};


template<class T>
void createLayoutFor(std::string path, std::string outPath, std::unordered_map<Argument, double>& settings) {
	switch ((Method)(int)settings[ARG_METHOD]) {
	case METHOD_NNPNET:
	{
		Graph<float> g(settings[ARG_DIMENSIONS]);
		NNPNET nnpnet;
		nnpnet.perplexity = settings[ARG_PERP];
		nnpnet.gpu = settings[ARG_GPU] > 0;
		nnpnet.useFloats = settings[ARG_USE_FLOAT] > 0;
		nnpnet.theta = settings[ARG_THETA];
		nnpnet.subgraphPoints = settings[ARG_SUBGRAPH_SIZE];
		nnpnet.pmdsPivots = settings[ARG_PMDS_PIVOTS];
		nnpnet.trainingEpochs = settings[ARG_TRAINING_EPOCHS];
		nnpnet.batchSize = settings[ARG_BATCH_SIZE];

		g.loadFromFile(path);
		TIME(nnpnet.run(g, nullptr, settings[ARG_EMBEDDING_SIZE]);
		if (settings[ARG_SMOOTH] >= 1) {
			SmoothingFunctions::Laplacian(g, settings[ARG_SMOOTH]);
		}, "NNP-NET");
		std::cout << "Saving to " << outPath << "\n";
		g.saveToVNA(outPath);
	}
		break;
	case METHOD_PMDS:
	{
		Graph<T> g(settings[ARG_DIMENSIONS]);
		g.loadFromFile(path);
		PivotMDS<T> pmds;
		pmds.setNumberOfPivots(settings[ARG_PMDS_PIVOTS]);
		TIME(pmds.call(g), "PMDS");
		std::cout << "Saving to " << outPath << "\n";
		g.saveToVNA(outPath);
	}
		break;
	case METHOD_TSNET:
	{
		Graph<double> g(settings[ARG_DIMENSIONS]);
		g.loadFromFile(path);
		TSNET<double> tsnet;
		tsnet.perp = settings[ARG_PERP];
		TIME(tsnet.tsNET(g, settings[ARG_THETA]), "tsNET");
		std::cout << "Saving to " << outPath << "\n";
		g.saveToVNA(outPath);
	}
		break;
	case METHOD_TSNETSTAR:
	{
		Graph<double> g(settings[ARG_DIMENSIONS]);
		g.loadFromFile(path);
		TSNET<double> tsnet;
		tsnet.perp = settings[ARG_PERP];
		TIME(tsnet.tsNETStar(g, settings[ARG_THETA]), "tsNET*");
		std::cout << "Saving to " << outPath << "\n";
		g.saveToVNA(outPath);
	}
	break;
	}
}

void printSettings(std::unordered_map<Argument, double>& settings) {
	std::cout << "Output dimensions: " << settings[ARG_DIMENSIONS] << "\n";
	std::cout << "Method used: ";
	switch((Method)(int)settings[ARG_METHOD]) {
	case METHOD_TSNET: std::cout << "tsNET\n";
		std::cout << "tsNET* settings: \n"
			<< "\tPerplexity: " << settings[ARG_PERP] << "\n"
			<< "\tTheta: " << settings[ARG_THETA] << "\n";
		break;
	case METHOD_TSNETSTAR: std::cout << "tsNET*\n";
		std::cout << "tsNET* settings: \n" 
			<< "\tPerplexity: " << settings[ARG_PERP] << "\n"
			<< "\tTheta: " << settings[ARG_THETA] << "\n";
		break;
	case METHOD_NNPNET: std::cout << "NNP-NET\n";
		std::cout << "NNP-NET settings: \n" 
			<< "\tPerplexity: " << settings[ARG_PERP] << "\n"
			<< "\tSubgraph size: " << (int)settings[ARG_SUBGRAPH_SIZE] << "\n"
			<< "\tEmbedding size: " << (int)settings[ARG_EMBEDDING_SIZE] << "\n"
			<< "\tTheta: " << settings[ARG_THETA] << "\n"
			<< "\tSmoothing passes: " << settings[ARG_SMOOTH] << "\n"
			<< "\tPMDS Pivots: " << settings[ARG_PMDS_PIVOTS] << "\n"
			<< "\tBatch size: " << settings[ARG_BATCH_SIZE] << "\n"
			<< "\tTraining epochs: " << settings[ARG_TRAINING_EPOCHS] << "\n"
			<< "\tUses gpu: " << (settings[ARG_GPU] == 0? "No" : "Yes") << "\n"
			<< "\tUses float precision: " << (settings[ARG_USE_FLOAT] == 0 ? "Double" : "Float") << "\n";

		break;
	case METHOD_PMDS: std::cout << "PMDS\n";
		std::cout << "tsNET* settings: \n" 
			<< "\tPMDS Pivots: " << settings[ARG_PMDS_PIVOTS] << "\n"
			<< "\tUses float precision: " << (settings[ARG_USE_FLOAT] == 0 ? "Double" : "Float") << "\n";
		break;
	}
	
}

int main(int argc, char* argv[])
{
	py::scoped_interpreter guard{};
	Threadpool::initialize();

	std::string path;
	std::string outPath;

	std::unordered_map<Argument, double> settings;
	settings[ARG_SMOOTH] = 3;
	settings[ARG_THETA] = 0.25;
	settings[ARG_GPU] = 0;
	settings[ARG_PERP] = 40;
	settings[ARG_METHOD] = METHOD_NNPNET;
	settings[ARG_EMBEDDING_SIZE] = 50;
	settings[ARG_SUBGRAPH_SIZE] = 10000;
	settings[ARG_PMDS_PIVOTS] = 250;
	settings[ARG_DIMENSIONS] = 2;
	settings[ARG_BATCH_SIZE] = 64;
	settings[ARG_TRAINING_EPOCHS] = 40;
	settings[ARG_USE_FLOAT] = 1;


	if (argc == 1) {
		std::cout << "Enter path\n";
		std::cin >> path;
	}
	else {
		int i = 1;
		if (argMap.count(argv[1]) == 0) {
			path = argv[1];
			i++;
		}
		else {
			std::cout << "Enter path\n";
			std::cin >> path;
		}
		for (; i < argc; i++) {
			if (argMap.count(argv[i]) == 0) {
				std::cout << "Invalid argument " << argv[i] << "\n";
				continue;
			}
			Argument setting = argMap[argv[i]];
			i++;
			if (i == argc || argMap.count(argv[i]) > 0) {
				i--;
				std::cout << "Value missing for argument " << argv[i] << "\n";
				continue;
			}
			std::string argS = std::string(argv[i]);
			switch (setting) {
				case ARG_SMOOTH:
				case ARG_THETA:
				case ARG_PERP:
				case ARG_EMBEDDING_SIZE:
				case ARG_SUBGRAPH_SIZE:
				case ARG_DIMENSIONS:
				case ARG_BATCH_SIZE:
				case ARG_TRAINING_EPOCHS:
				case ARG_PMDS_PIVOTS:
					settings[setting] = std::stod(argv[i]);
					break;
				case ARG_OUTPATH:
					outPath = argv[i];
					break;
				case ARG_GPU:
				case ARG_USE_FLOAT:
					settings[setting] = (argS == "t" || argS == "true" || argS == "T" || argS == "True" || argS == "TRUE" || argS == "y" || argS == "Y" || argS == "yes" || argS == "Yes" || argS == "YES" || argS == "1") ? 1 : 0;
					break;
				case ARG_METHOD:
					if (argS == "tsNET" || argS == "tsnet" || argS == "TSNET") {
						settings[setting] = METHOD_TSNET;
					}
					else if (argS == "tsNET*" || argS == "tsnet*" || argS == "TSNET*") {
						settings[setting] = METHOD_TSNETSTAR;
					}
					else if (argS == "pmds" || argS == "PMDS" || argS == "Pmds" || argS == "PivotMDS" || argS == "pivotmds" || argS == "pivotMDS") {
						settings[setting] = METHOD_PMDS;
					}
					else if (argS == "NNP-NET" || argS == "NNPNET" || argS == "nnp-net" || argS == "NNP-net" || argS == "nnp-NET" || argS == "nnpnet" || argS == "nnpNet" || argS == "nnpNET") {
						settings[setting] = METHOD_NNPNET;
					}
					else {
						std::cout << "Embedding method " << argv[i] << " not recognized, defaulting to NNP-NET. Use either:\nNNP-NET\nPMDS\ntsNET\ntsNET*\n\n";
					}
					break;

			}
			
		}
	}

	printSettings(settings);

	if (path[path.size() - 4] == '.') {
		if (outPath == "") {
			outPath = path.substr(0, path.size() - 4) + "_out.vna";
		}
		if (!(outPath[outPath.size() - 3] == 'v' && outPath[outPath.size() - 2] == 'n' && outPath[outPath.size() - 1] == 'a')) {
			outPath += ".vna";
		}
		if (settings[ARG_USE_FLOAT]) {
			createLayoutFor<float>(path, outPath, settings);
		}
		else {
			createLayoutFor<double>(path, outPath, settings);
		}
		
	}
	else {
		for (const auto& entry : fs::directory_iterator(path)) {
			std::string p = entry.path().string();
			auto splittedPath = Utils::split(p, '\\');
			if (splittedPath.size() == 1) {
				splittedPath = Utils::split(p, '/');
			}
			std::string fileName = splittedPath[splittedPath.size() - 1];

			std::cout << "\nCurrent file: " << p << "\n";

			std::string oPath;
			if (outPath == "") {
				oPath = p.substr(0, p.size() - 4) + "_out.vna";
			}
			else {
				oPath = outPath;
				if (oPath[oPath.size() - 1] != '\\' || oPath[oPath.size() - 1] != '/') {
					oPath += '/';
				}
				oPath += fileName.substr(0, fileName.length() - 4) + ".vna";
			}

			if (settings[ARG_USE_FLOAT]) {
				createLayoutFor<float>(p, oPath, settings);
			}
			else {
				createLayoutFor<double>(p, oPath, settings);
			}
		}
	}

	return 0;
}
