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
	{"-d", ARG_OUTPATH},
	{"--dimensions", ARG_OUTPATH},
};


void createLayoutFor(std::string path, std::string outPath, std::unordered_map<Argument, double>& settings) {
	switch ((Method)(int)settings[ARG_METHOD]) {
	case METHOD_NNPNET:
	{
		Graph<float> g(settings[ARG_DIMENSIONS]);
		NNPNET nnpnet;
		nnpnet.perplexity = settings[ARG_PERP];
		nnpnet.gpu = settings[ARG_GPU] > 0;
		nnpnet.theta = settings[ARG_THETA];
		nnpnet.subgraphPoints = settings[ARG_SUBGRAPH_SIZE];
		nnpnet.pmdsPivots = settings[ARG_PMDS_PIVOTS];

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
		Graph<float> g(settings[ARG_DIMENSIONS]);
		PivotMDS<float> pmds;
		pmds.setNumberOfPivots(settings[ARG_PMDS_PIVOTS]);
		TIME(pmds.call(g), "PMDS");
		std::cout << "Saving to " << outPath << "\n";
		g.saveToVNA(outPath);
	}
		break;
	case METHOD_TSNET:
	{
		Graph<double> g(settings[ARG_DIMENSIONS]);
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
		TSNET<double> tsnet;
		tsnet.perp = settings[ARG_PERP];
		TIME(tsnet.tsNETStar(g, settings[ARG_THETA]), "tsNET*");
		std::cout << "Saving to " << outPath << "\n";
		g.saveToVNA(outPath);
	}
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
			if (i == argc || argMap.count(argv[i]) == 0) {
				i--;
				std::cout << "Value missing for argument " << argv[i] << "\n";
				continue;
			}
			switch (setting) {
				case ARG_SMOOTH:
				case ARG_THETA:
				case ARG_PERP:
				case ARG_EMBEDDING_SIZE:
				case ARG_SUBGRAPH_SIZE:
				case ARG_DIMENSIONS:
				case ARG_PMDS_PIVOTS:
					settings[setting] = std::stod(argv[i]);
					break;
				case ARG_OUTPATH:
					outPath = argv[i];
					break;
				case ARG_GPU:
					settings[setting] = (argv[i] == "t" || argv[i] == "true" || argv[i] == "T" || argv[i] == "True" || argv[i] == "TRUE" || argv[i] == "y" || argv[i] == "Y" || argv[i] == "yes" || argv[i] == "Yes" || argv[i] == "YES" || argv[i] == "1");
					break;
				case ARG_METHOD:
					if (argv[i] == "tsNET" || argv[i] == "tsnet" || argv[i] == "TSNET") {
						settings[setting] = METHOD_TSNET;
					}
					else if (argv[i] == "tsNET*" || argv[i] == "tsnet*" || argv[i] == "TSNET*") {
						settings[setting] = METHOD_TSNETSTAR;
					}
					else if (argv[i] == "pmds" || argv[i] == "PMDS" || argv[i] == "Pmds" || argv[i] == "PivotMDS" || argv[i] == "pivotmds" || argv[i] == "pivotMDS") {
						settings[setting] = METHOD_PMDS;
					}
					else if (argv[i] == "NNP-NET" || argv[i] == "NNPNET" || argv[i] == "nnp-net" || argv[i] == "NNP-net" || argv[i] == "nnp-NET" || argv[i] == "nnpnet" || argv[i] == "nnpNet" || argv[i] == "nnpNET") {
						settings[setting] = METHOD_NNPNET;
					}
					else {
						std::cout << "Embedding method " << argv[i] << " not recognized, defaulting to NNP-NET. Use either:\nNNP-NET\nPMDS\ntsNET\ntsNET*\n\n";
					}
					break;

			}
			
		}
	}



	if (path[path.size() - 4] == '.') {
		if (outPath == "") {
			outPath = path.substr(0, path.size() - 4) + "_out.vna";
		}
		if (!(outPath[outPath.size() - 3] == 'v' && outPath[outPath.size() - 2] == 'n' && outPath[outPath.size() - 1] == 'a')) {
			outPath += ".vna";
		}
		createLayoutFor(path, outPath, settings);
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

			createLayoutFor(p, oPath, settings);
		}
	}

	return 0;
}
