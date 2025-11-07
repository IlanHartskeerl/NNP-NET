#pragma once

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

#include "Utils.h"

namespace NNPNet {

	template<typename T>
	struct Edge {
		Edge(int other, T weight) {
			this->other = other;
			this->weight = weight;
		}

		int other;
		T weight;
	};

	template<class T>
	class Graph {
	public:

		Graph(int outputDimensions) : edges() { this->outputDim = outputDimensions; };

		/// <summary>
		/// Copy constructor for when the output type does not match
		/// </summary>
		template<class P>
		Graph(Graph<P>& g) {
			nodeCount = g.nodeCount;
			outputDim = g.outputDim;
			weighted = g.weighted;
			edges.resize(g.edges.size());
			int i = 0;
			for (auto& es : g.edges) {
				for (auto e : es) {
					edges[i].push_back(Edge<T>(e.other, (T)e.weight));
				}
				i++;
			}
			Y = (T*)malloc(((size_t)nodeCount) * ((size_t)outputDim) * sizeof(T));
			if (g.nodeNames.size() > 0) {
				nodeNames.resize(g.nodeNames.size());
				memcpy(nodeNames.data(), g.nodeNames.data(), sizeof(int) * g.nodeNames.size());
				for (int i = 0; i < nodeCount; i++) {
					namesToId[nodeNames[i]] = i;
				}
			}
		}

		~Graph() {
			free(Y);
		}

		void gridGraph(int size) {
			nodeCount = size * size;
			edges.resize(nodeCount);
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					if (i > 0) {
						edges[i * size + j].push_back(Edge((i - 1) * size + j, 1));
						edges[(i - 1) * size + j].push_back(Edge(i * size + j, 1));
					}
					if (j > 0) {
						edges[i * size + j].push_back(Edge(i * size + (j - 1), 1));
						edges[i * size + (j - 1)].push_back(Edge(i * size + j, 1));
					}

				}
			}
			Y = (T*)malloc(nodeCount * outputDim * sizeof(T));
		};

		void getDistances(int source, T* distances) {
			if (weighted) {
				dijkstra(source, distances);
			}
			else {
				bfs(source, distances);
			}
		};

		bool checkFullyConnected() {
			T* dist = (T*)malloc(sizeof(T) * nodeCount);
			for (int i = 0; i < nodeCount; i++) {
				dist[i] = -1;
			}
			bfs(0, dist);
			for (int i = 0; i < nodeCount; i++) {
				if (dist[i] == -1) {
					return false;
				}
			}
			return true;
		}

		void dijkstra(int source, T* distances) {
			bool* visited = (bool*)malloc(nodeCount * sizeof(bool));
			for (int i = 0; i < nodeCount; i++) {
				visited[i] = false;
			}
			distances[source] = 0;

			std::map<T, std::vector<int>> stack;
			visited[source] = true;
			bool addExtraEdges = edges[source].size() != nodeCount - 1;
			for (auto& e : edges[source]) {
				stack[e.weight].push_back(e.other);
			}
			int pointsLeft = nodeCount - 1;
			while (stack.size() > 0 && pointsLeft > 0) {

				while (stack.size() > 0 && stack.begin()->second.size() == 0) {
					auto key = stack.begin()->first;
					stack.erase(key);
				}
				if (stack.size() == 0) {
					break;
				}
				auto lowest = stack.begin()->second[stack.begin()->second.size() - 1];
				auto length = stack.begin()->first;
				
				stack.begin()->second.pop_back();
				if (visited[lowest]) {
					continue;
				}
				pointsLeft--;
				visited[lowest] = true;
				distances[lowest] = length;
				if (!addExtraEdges) continue;
				for (auto& e : edges[lowest]) {
					if (visited[e.other]) continue;
					stack[length + e.weight].push_back(e.other);
				}
			}
			free(visited);
		};

		int rnn(int source, int* nodes, T maxDistance) {
			if (!weighted && maxDistance == 2) {
				return rnnUnweightedR2(source, nodes);
			}
			return rnnWeighted(source, nodes, maxDistance);
		}

		int rnnWeighted(int source, int* nodes, T maxDistance) {

			std::unordered_set<int> visited;

			std::map<T, std::vector<int>> stack;
			visited.insert(source);
			bool addExtraEdges = edges[source].size() != nodeCount - 1;
			for (auto& e : edges[source]) {
				if (e.weight <= maxDistance) {
					stack[e.weight].push_back(e.other);
				}
			}
			int nodeC = 0;
			int pointsLeft = nodeCount - 1;
			while (stack.size() > 0 && pointsLeft > 0) {
				while (stack.size() > 0 && stack.begin()->second.size() == 0) {
					auto key = stack.begin()->first;
					stack.erase(key);
				}
				if (stack.size() == 0) {
					break;
				}
				auto lowest = stack.begin()->second[stack.begin()->second.size() - 1];
				auto length = stack.begin()->first;
				stack.begin()->second.pop_back();
				if (visited.count(lowest) > 0) {
					continue;
				}
				pointsLeft--;
				visited.insert(lowest);
				nodes[nodeC++] = lowest;
				if (!addExtraEdges) continue;
				for (auto& e : edges[lowest]) {
					if (visited.count(e.other) > 0 || length + e.weight > maxDistance) continue;
					stack[length + e.weight].push_back(e.other);
				}
			}
			return nodeC;
		};

		int rnnUnweightedR2(int source, int* nodes) {
			std::unordered_set<int> visited;

			for (auto& e1 : edges[source]) {
				if (e1.other == source) continue;
				visited.insert(e1.other);
				for (auto& e2 : edges[e1.other]) {
					visited.insert(e2.other);
				}
			}
			int count = 0;
			for (int i : visited) {
				if (i == source) continue;
				nodes[count++] = i;
			}
			return count;
		}

		void knn(int source, int* nodes, T* distances, int k) {
			std::unordered_set<int> visited;

			std::map<T, std::vector<int>> stack;
			visited.insert(source);
			bool addExtraEdges = edges[source].size() != nodeCount - 1;
			for (auto& e : edges[source]) {
				stack[e.weight].push_back(e.other);
			}
			int pointsLeft = k;
			int i = 0;
			while (stack.size() > 0 && pointsLeft > 0) {

				while (stack.size() > 0 && stack.begin()->second.size() == 0) {
					auto key = stack.begin()->first;
					stack.erase(key);
				}
				if (stack.size() == 0) {
					break;
				}
				auto lowest = stack.begin()->second[stack.begin()->second.size() - 1];
				auto length = stack.begin()->first;
				stack.begin()->second.pop_back();
				if (visited.count(lowest) > 0) {
					continue;
				}
				pointsLeft--;
				visited.insert(lowest);
				distances[i] = length;
				nodes[i++] = lowest;
				if (!addExtraEdges) continue;
				for (auto& e : edges[lowest]) {
					if (visited.count(e.other) > 0) continue;
					stack[length + e.weight].push_back(e.other);
				}
			}
		};

		void bfs(int source, T* distances) {
			for (int i = 0; i < nodeCount; i++) {
				distances[i] = -1;
			}

			int pointsLeft = nodeCount;
			std::vector<int> nextList;
			std::vector<int> curList;

			curList.push_back(source);
			distances[source] = 0;
			int curDist = 1;
			while (curList.size() > 0) {
				nextList.clear();

				for (int s : curList) {
					for (const auto& edge : edges[s]) {
						if (distances[edge.other] != -1) continue;
						distances[edge.other] = curDist;
						nextList.push_back(edge.other);
					}
				}
				curDist++;
				std::swap(curList, nextList);
			}

		};

		void loadFromFile(std::string path) {
			if (path[path.size() - 3] == 'm' && path[path.size() - 2] == 't' && path[path.size() - 1] == 'x') {
				loadFromMTX(path);
			}
			else if (path[path.size() - 3] == 'v' && path[path.size() - 2] == 'n' && path[path.size() - 1] == 'a') {
				loadFromVNA(path);
			}
			else if (path[path.size() - 3] == 'd' && path[path.size() - 2] == 'o' && path[path.size() - 1] == 't') {
				loadFromDOT(path);
			}
			else if (path[path.size() - 3] == 't' && path[path.size() - 2] == 'x' && path[path.size() - 1] == 't') {
				loadFromTXT(path);
			}
			else {
				std::cout << "File does not have a extention that is supported\nProvide either a .mtx, .vna or .dot file file\n";
			}
		};

		void loadFromVNA(std::string path) {
			std::ifstream infile(path);

			std::string line;

			int node = 0;
			std::unordered_map<std::string, int> nodes;
			// skip first 2 lines
			std::getline(infile, line); std::getline(infile, line);

			while (std::getline(infile, line))
			{
				if (line[0] == '*') break;
				nodes[Utils::getFirstTokenInString(line)] = node++;
			}
			nodeCount = nodes.size();
			edges.resize(nodeCount);
			Y = (T*)malloc(nodeCount * outputDim * sizeof(T));
			if (line == "*Node properties") {
				std::getline(infile, line);
				if (line == "ID x y") {
					while (std::getline(infile, line))
					{
						if (line[0] == '*') break;
						std::vector<std::string> splitted = Utils::split(line);
						Y[nodes[splitted[0]] * outputDim] = std::stof(splitted[1]);
						Y[nodes[splitted[0]] * outputDim + 1] = std::stof(splitted[2]);
					}
				}
				else {
					// Skip
					while (std::getline(infile, line))
					{
						if (line[0] == '*') break;
					}
				}
			}
			std::getline(infile, line);
			while (std::getline(infile, line))
			{
				if (line.size() <= 4) break;
				std::vector<std::string> splitted = Utils::split(line);
				int node0 = nodes[splitted[0]];
				int node1 = nodes[splitted[1]];
				if (node0 == node1) continue;
				T weight = std::stof(splitted[2]);
				if (weight != 1) {
					weighted = true;
				}
				edges[node0].push_back({ node1, weight });
				edges[node1].push_back({ node0, weight });
			}

		};

		void loadFromMTX(std::string path) {
			std::ifstream infile(path);

			std::string line;
			// Skip first line, as it has the sizes.
			while (std::getline(infile, line))
			{
				if (line[0] == '%') continue;
				break;
			}
			T highestWeight = 0;
			while (std::getline(infile, line))
			{
				if (line.size() <= 2) continue;
				if (line[0] == '%') continue;
				auto splitted = Utils::split(line, ' ');
				if (splitted.size() < 2) continue;
				T w = 1;
				if (splitted.size() == 3) {
					w = std::stod(splitted[2]);
					if (w > highestWeight) {
						highestWeight = w;
					}
					weighted = true;
				}
				int n0 = std::stoi(splitted[0]) - 1;
				int n1 = std::stoi(splitted[1]) - 1;
				if (n0 == n1) continue;
				int biggest = n0 < n1 ? n1 + 1 : n0 + 1;
				if (edges.size() < biggest) {
					edges.resize(biggest);
				}
				bool exists = false;
				for (auto e : edges[n0]) {
					if (e.other == n1) {
						exists = true;
						break;
					}
				}
				if (!exists) {
					edges[n0].push_back(Edge(n1, w));
					edges[n1].push_back(Edge(n0, w));
				}
			}
			nodeCount = edges.size();
			Y = (T*)malloc(nodeCount * outputDim * sizeof(T));

			if (weighted) {
				for (auto& e : edges) {
					for (auto& edge : e) {
						edge.weight /= highestWeight;
					}
				}
			}
		};

		void loadFromDOT(std::string path)
		{
			std::ifstream infile(path);

			nodeCount = 0;
			int lastNode = -1;
			int min = 0;

			std::string line;
			std::unordered_map<int, std::pair<T, T >> positions;
			while (std::getline(infile, line))
			{
				if (line.size() <= 1) continue;
				auto splitted = Utils::split(line.substr(1));
				if (line[1] >= '0' && line[1] <= '9' && splitted.size() > 1 && splitted[1][0] == '-') {
					// Edge
					int p0 = std::stoi(splitted[0]) - min;
					int p1 = std::stoi(Utils::split(splitted[2], '\t')[0]) - min;

					if (edges.size() <= p1) {
						edges.resize(p1 + 1);
					}
					if (edges.size() <= p0) {
						edges.resize(p0 + 1);
					}
					edges[p0].push_back(Edge<T>(p1, 1));
					edges[p1].push_back(Edge<T>(p0, 1));
					lastNode = -1;
				}
				else if (line[1] >= '0' && line[1] <= '9') {
					int node = std::stoi(Utils::split(splitted[0], '\t')[0]);
					if (nodeCount == 0) {
						min = node;
					}
					node -= min;
					lastNode = node;

					if (nodeCount <= node) {
						nodeCount = node + 1;
					}

					for (std::string& s : splitted) {
						if (s.substr(0, 3) == "pos") {
							auto pos = Utils::split(Utils::split(s, '\"')[1], ',');

							positions[node] = std::pair<T, T>(std::stof(pos[0]), std::stof(pos[1]));
							break;
						}
					}
				}
				else if (lastNode != -1 && line[2] == 'p' && line[3] == 'o' && line[4] == 's') {
					auto pos = Utils::split(Utils::split(line, '\"')[1], ',');

					positions[lastNode] = std::pair<T, T>(std::stof(pos[0]), std::stof(pos[1]));
				}
			}
			Y = (T*)malloc(nodeCount * outputDim * sizeof(T));
			for (int i = 0; i < nodeCount; i++) {
				Y[i * 2] = positions[i].first;
				Y[i * 2 + 1] = positions[i].second;
			}
			normalize();
		}

		void loadFromTXT(std::string path) 
		{
			std::ifstream infile(path);

			std::string line;

			std::getline(infile, line);
			nodeCount = std::stoi(Utils::split(line, ' ')[0]);
			edges.resize(nodeCount);
			while (std::getline(infile, line))
			{
				if (line.size() < 2) continue;
				auto splitted = Utils::split(line);

				int e0 = std::stoi(splitted[0]);
				int e1 = std::stoi(splitted[1]);

				if (e0 < 0 || e1 < 0) continue;

				edges[e0].push_back(Edge<T>(e1, 1));
				edges[e1].push_back(Edge<T>(e0, 1));
			}
			Y = (T*)malloc(nodeCount * outputDim * sizeof(T));
		}

		void loadTimeSeries(std::string path, bool onlyAdds = false) {
			std::ifstream infile(path);

			std::string line;
			namesToId.clear();
			for (int i = 0; i < nodeCount; i++) {
				namesToId[nodeNames[i]] = i;
			}

			while (std::getline(infile, line)) {
				auto split = Utils::split(line, ' ');
				if (split[0] == "dn" && !onlyAdds) {
					int node = namesToId[std::stoi(split[1])];
					// Remove edges that reference this node
					for (auto e : edges[node]) {
						for (int i = 0; i < edges[e.other].size(); i++) {
							if (edges[e.other][i].other == node) {
								edges[e.other][i] = edges[e.other][edges[e.other].size() - 1];
								edges[e.other].pop_back();
								break;
							}
						}
					}
					int swap = nodeCount - 1;
					namesToId[nodeNames[swap]] = node;
					edges[node].clear();
					edges[node] = edges[swap];
					edges.pop_back();
					nodeNames[node] = nodeNames[swap];
					nodeNames.pop_back();

					// Fix edges of moved node
					for (auto e : edges[node]) {
						for (int i = 0; i < edges[e.other].size(); i++) {
							if (edges[e.other][i].other == swap) {
								edges[e.other][i].other = node;
								break;
							}
						}
					}

					nodeCount--;
				}
				else if (split[0] == "an") {
					nodeNames.push_back(std::stoi(split[1]));
					namesToId[std::stoi(split[1])] = nodeCount;
					nodeCount++;
					edges.resize(nodeCount);
				}
				else if (split[0] == "ae") {
					int a = namesToId[std::stoi(split[1])];
					int b = namesToId[std::stoi(split[2])];
					float w = 1;
					if (split.size() > 3) {
						w = std::stof(split[3]);
					}
					edges[a].push_back(Edge<T>(b, w));
					edges[b].push_back(Edge<T>(a, w));
				}
				else if (split[0] == "cw") {
					int a = namesToId[std::stoi(split[1])];
					int b = namesToId[std::stoi(split[2])];
					float w = std::stof(split[3]);
					for (int i = 0; i < edges[a].size(); i++) {
						if (edges[a][i].other == b) {
							edges[a][i].weight = w;
							break;
						}
					}
					for (int i = 0; i < edges[b].size(); i++) {
						if (edges[b][i].other == a) {
							edges[b][i].weight = w;
							break;
						}
					}
				}
			}
			free(Y);
			Y = (T*)malloc(sizeof(T) * outputDim * nodeCount);

		}

		bool* checkNodesInAllTimeSeries(std::string path) {
			std::vector<char> inAll;
			inAll.resize(nodeCount);
			
			for (int j = 0; j < nodeCount; j++) {
				inAll[j] = true;
			}

			int i = 1;
			std::string tsdPath = path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
			while (std::filesystem::exists(tsdPath)) {
				std::ifstream infile(tsdPath);
				
				std::string line;

				while (std::getline(infile, line)) {
					if (line.size() > 1) {
						auto split = Utils::split(line, ' ');
						if (split[0] == "dn") {
							inAll[std::stoi(split[1])] = false;
						}
					}
				}

				i++;
				tsdPath = path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
			}

			bool* res = (bool*)malloc(sizeof(bool) * inAll.size());
			memcpy(res, inAll.data(), sizeof(bool) * inAll.size());
			return res;
		}

		void onlyConnectedFrom(int id) {
			T* distances = (T*)malloc(nodeCount * sizeof(T));

			for (int i = 0; i < nodeCount; i++) {
				distances[i] = -1;
			}
			bfs(id, distances);
			for (int i = nodeCount - 1; i >= 0; i--) {
				if (distances[i] == -1) {
					for (auto e : edges[i]) {
						for (int j = 0; j < edges[e.other].size(); j++) {
							if (edges[e.other][j].other == i) {
								edges[e.other][j] = edges[e.other][edges[e.other].size() - 1];
								edges[e.other].pop_back();
								break;
							}
						}
					}
					edges[i].clear();
					int swap = nodeCount - 1;
					nodeNames[i] = nodeNames[swap];
					nodeNames.pop_back();
					edges[i].clear();
					edges[i] = edges[swap];
					for (auto e : edges[i]) {
						for (int j = 0; j < edges[e.other].size(); j++) {
							if (edges[e.other][j].other == swap) {
								edges[e.other][j].other = i;
								break;
							}
						}
					}
					edges.pop_back();
					nodeCount--;
				}
			}
			free(Y);
			Y = (T*)malloc(sizeof(T) * outputDim * nodeCount);
			namesToId.clear();
			for (int i = 0; i < nodeCount; i++) {
				namesToId[nodeNames[i]] = i;
			}
		}


		void saveToVNA(std::string path) {
			if (nodeNames.size() == 0) {
				fillNodeNames();
			}

			std::ofstream file;
			file.open(path);
			file << "*Node data\nID\n";
			for (int i = 0; i < nodeCount; i++) {
				file << std::to_string(nodeNames[i]) << '\n';
			}
			if (outputDim == 2) {
				file << "*Node properties\nID x y\n";
				for (int i = 0; i < nodeCount; i++) {
					file << nodeNames[i] << " " << Y[i * 2] << " " << Y[i * 2 + 1] << "\n";
				}
			}
			else if (outputDim == 3) {
				file << "*Node properties\nID x y z\n";
				for (int i = 0; i < nodeCount; i++) {
					file << nodeNames[i] << " " << Y[i * 3] << " " << Y[i * 3 + 1] << " " << Y[i * 3 + 2] << "\n";
				}
			}
			else {
				file << "*Node properties\nID";
				for (int i = 0; i < outputDim; i++) {
					file << " d" << std::to_string(i);
				}
				file << "\n";
				for (int i = 0; i < nodeCount; i++) {
					file << nodeNames[i];
					for (int j = 0; j < outputDim; j++) {
						file << " " << Y[i * outputDim + j];
					}
					file << "\n";
				}
			}
			file << "*Tie data\nfrom to strength\n";
			int i = 0;
			for (auto& edge : edges) {
				for (auto to : edge) {
					if (to.other < i) continue; // Skip duplicated edges
					file << nodeNames[i] << " " << nodeNames[to.other] << " " << to.weight << '\n';
				}
				i++;
			}

			file.close();
		};

		void normalize() {
			T* min, * max;
			min = (T*)malloc(sizeof(T) * ((size_t)outputDim));
			max = (T*)malloc(sizeof(T) * ((size_t)outputDim));
			for (int i = 0; i < outputDim; i++) {
				min[i] = Y[i];
				max[i] = Y[i];
			}
			for (size_t i = 1; i < nodeCount; i++) {
				for (size_t d = 0; d < outputDim; d++) {
					if (min[d] > Y[i * ((size_t)outputDim) + d]) {
						min[d] = Y[i * ((size_t)outputDim) + d];
					}
					if (max[d] < Y[i * ((size_t)outputDim) + d]) {
						max[d] = Y[i * ((size_t)outputDim) + d];
					}
				}
			}
			T maxSize = max[0] - min[0];
			for (int i = 1; i < outputDim; i++) {
				if (max[i] - min[i] > maxSize) {
					maxSize = max[i] - min[i];
				}
			}
			for (size_t i = 0; i < nodeCount; i++) {
				for (size_t d = 0; d < outputDim; d++) {
					Y[i * ((size_t)outputDim) + d] = (Y[i * ((size_t)outputDim) + d] - min[d]) / maxSize;
				}
			}
		};

		T getDistanceBetween(int n0, int n1) {
			T tot = 0;
			for (int i = 0; i < outputDim; i++) {
				T d = Y[n0 * outputDim + i] - Y[n1 * outputDim + i];
				tot += d * d;
			}
			return sqrt(tot);
		}

		void fillNodeNames() {
			nodeNames.clear();
			for (int i = 0; i < nodeCount; i++) {
				nodeNames.push_back(i);
			}
		}

		int nodeCount = -1, outputDim;
		std::vector<std::vector<Edge<T>>> edges;
		std::vector<int> nodeNames;
		std::unordered_map<int, int> namesToId;

		T* Y = nullptr;

		bool weighted = false;

	};

};
