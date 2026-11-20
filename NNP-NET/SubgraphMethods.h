#include "Graph.h"
#include <random>

#define SUBG_VECTORIZATION

#define ALLOW_LESS_STRICT_MERGING
#define SET_ORDERING
//#define REVERSE_ORDER

//#define ASSC_SET_ORDER

namespace NNPNet {
	class SubgraphMethods {
	public:
		template <class T>
		static T createFastPivotSubgraph(Graph<T>& g, Graph<T>& outG, std::vector<int>& nodes, int subpoints) {
			// Data setup
			std::unordered_map<int, int> pivots;
			T* closest = (T*)malloc(g.nodeCount * sizeof(T));

			outG.nodeCount = subpoints;
			outG.edges.resize(subpoints);
			outG.Y = (T*)malloc(subpoints * outG.outputDim * sizeof(T));

			// First pivot
			g.getDistances(0, closest);
			pivots[0] = 0;
			nodes.push_back(0);

			T lastPivotDistance = -1;
			int posInList = 1;

			// All other pivots
			for (int i = 1; i < subpoints; i++) {
				// Choose next pivot
				int pivot = -1;
				if (lastPivotDistance != -1) {
#ifdef SUBG_VECTORIZATION
					__m256 lpd8 = _mm256_set1_ps(lastPivotDistance);
					for (posInList; posInList < g.nodeCount - 7; posInList += 8) {
						if (_mm256_movemask_ps(_mm256_cmp_ps(lpd8, *(__m256*)(closest + posInList), 0))) {
							for (posInList; posInList < g.nodeCount; posInList++) {
								if (closest[posInList] == lastPivotDistance) {
									pivot = posInList;
									break;
								}
							}
							break;
						}
					}
					if (pivot == -1) {
#endif
						for (posInList; posInList < g.nodeCount; posInList++) {
							if (closest[posInList] == lastPivotDistance) {
								pivot = posInList;
								break;
							}
						}
#ifdef SUBG_VECTORIZATION
					}
#endif
				}
				// Get new pivot distance
				if (pivot == -1) {
					std::function < std::pair<T, int>(int, int)> f = [closest](int begin, int end) -> std::pair<T, int>
						{
							int pivot = 0;
							T distance = 0;
							for (int n = begin; n < end; n++) {
								if (distance < closest[n]) {
									distance = closest[n];
									pivot = n;
								}
							}
							return std::pair<T, int>{ distance, pivot };
						};
					auto pivotDist = Threadpool::divideWork(
						f, g.nodeCount,
						(std::function<std::pair<T, int>(std::pair<T, int>, std::pair<T, int>)>)[](std::pair<T, int> a, std::pair<T, int> b)->std::pair<T, int> {return a.first > b.first ? a : b; });
					lastPivotDistance = pivotDist.first;
					pivot = pivotDist.second;
					posInList = 1; // One because 0 is always the first pivot
				}
				pivots[pivot] = i;
				nodes.push_back(pivot);

				// Add the pivot
				g.updateDistances(pivot, closest);
			}

			// Get r for r-nn
			T r = 0;
			for (int n = 1; n < g.nodeCount; n++) {
				if (r < closest[n]) {
					r = closest[n];
				}
			}
			//r+=2;
			if (g.weighted) {
				r *= 2;
			}
			else {
				r *= 2;
				r++; // Not sure if trully needed, but a lot safer
			}

			// Create edges
			Threadpool::divideWork([&g, &outG, &nodes, r, &pivots](int begin, int end)
				{
					int* nodeIndices = (int*)malloc(g.nodeCount * sizeof(int));
					T* distances = (T*)malloc(g.nodeCount * sizeof(T));
					for (int j = begin; j < end; j++) {
						int p = nodes[j];
						int count = g.rnn(p, nodeIndices, r, distances);
						int a = pivots[p];
						for (int i = 0; i < count; i++) {
							if (pivots.count(nodeIndices[i]) > 0 && nodeIndices[i] != p) {
								int b = pivots[nodeIndices[i]];
								T w = distances[i];
								outG.edges[a].push_back(Edge<T>(b, w));
							}
						}
					}
					free(nodeIndices);
					free(distances);
				}, nodes.size());

			free(closest);

			return r;
		}


		template <class T>
		static void createFastSubnetwork(Graph<T>& inG, Graph<T>& outG, std::vector<int>& nodes, int subgraphPoints) {
			int target = subgraphPoints;
			Graph<T> temp(inG.outputDim);
			Graph<T> temp2(inG.outputDim);
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
			T reduction = 0;
			while (currentNodeCount > target && reduction < 0.95) {
				std::cout << currentNodeCount << "\n";
				if (currentIsFirst) {
					startPos = fastSubnetworkIteration(temp, temp2, tempNodes, tempNodes2, target, startPos);
					currentNodeCount = temp2.nodeCount;
					currentIsFirst = false;
					reduction = ((T)temp2.nodeCount) / ((T)temp.nodeCount);
				}
				else {
					startPos = fastSubnetworkIteration(temp2, temp, tempNodes2, tempNodes, target, startPos);
					currentNodeCount = temp.nodeCount;
					currentIsFirst = true;
					reduction = ((T)temp.nodeCount) / ((T)temp2.nodeCount);
				}
			}
			if (currentIsFirst) {
				if (currentNodeCount > target) {
					std::cout << "Does not collapse nicelly, reverting to pivot method to reduced graph\n";
					temp.weighted = true;
					createPivotSubnetwork(temp, outG, nodes, target);
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
					createPivotSubnetwork(temp2, outG, nodes, target);
				}
				else {
					outG = temp2;
					for (int i : tempNodes2) {
						nodes.push_back(i);
					}
				}
			}
			outG.Y = (T*)malloc(sizeof(T) * outG.nodeCount * outG.outputDim);
			std::cout << "Number of subgraph points reached: " << outG.nodeCount << "\n";
		};

		template <class T>
		static void createPivotSubnetwork(Graph<T>& g, Graph<T>& outG, std::vector<int>& nodes, int subgraphPoints) {
			int targetPoints = subgraphPoints;
			outG.Y = (T*)malloc(targetPoints * g.outputDim * sizeof(T));
			T* embedding = (T*)malloc(g.nodeCount * sizeof(T));
			int n = g.nodeCount;
			int pivot = 0;
			T* lowest = (T*)malloc(sizeof(T) * n);
			outG.edges.resize(targetPoints);
			for (int i = 0; i < targetPoints; i++) {
				g.getDistances(pivot, embedding);
				nodes.push_back(pivot);

				// Next pivot point
				if (i + 1 < targetPoints) {
					T highest = 0;
					if (i == 0) {
						std::memcpy(lowest, embedding, n * sizeof(T));
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
				T l = 9999999999;
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

		};


		static void ASSC(Graph<float>& inG, Graph<float>& outG, std::vector<int>& nodes, int subgraphPoints) {
			int target = subgraphPoints;
			Graph<float> temp(inG.outputDim);
			Graph<float> temp2(inG.outputDim);
			outG.weighted = inG.weighted;
			temp.weighted = inG.weighted;
			temp2.weighted = inG.weighted;

			std::vector<int> tempNodes;
			std::vector<int> tempNodes2;
			// fill temp nodes with all numbers
			tempNodes.resize(inG.nodeCount);
			for (int j = 0; j < inG.nodeCount; j++) {
				tempNodes[j] = j;
			}

			// --- Phase 1: Solar System Coarsening iterations ---
			// We keep a reference to the "previous" graph (still above target) so that
			// when an iteration overshoots below the target, we can fall back to the
			// old fastSubnetworkIteration method on that previous graph.

			// First iteration: previous graph is inG itself
			solarSystemIteration(inG, temp2, tempNodes, tempNodes2, target);
			firstIterCoarseningRate = (double)temp2.nodeCount / (double)inG.nodeCount;
			std::cout << temp2.nodeCount << "\n";

			if (temp2.nodeCount <= target && inG.nodeCount > target) {
				// First iteration already overshot — fall back to old method on inG
				std::cout << "Solar System overshot on first iteration, falling back to old coarsening on original graph\n";
				Graph<float> overshotGraph = temp2;
				std::vector<int> overshotNodes = tempNodes2;
				tempNodes.clear();
				tempNodes.resize(inG.nodeCount);
				for (int j = 0; j < inG.nodeCount; j++) {
					tempNodes[j] = j;
				}
				// Use temp/temp2 for the old method's ping-pong
				temp = Graph<float>(inG.outputDim);
				temp.weighted = inG.weighted;
				temp2 = Graph<float>(inG.outputDim);
				temp2.weighted = inG.weighted;
				tempNodes2.clear();

				int startPos = 0;
				startPos = fastSubnetworkIteration(inG, temp2, tempNodes, tempNodes2,
					target, startPos);
				int currentNodeCount = temp2.nodeCount;
				bool currentIsFirst = false;
				float reduction = 0;
				while (currentNodeCount > target && (reduction < 0.95)) {
					std::cout << currentNodeCount << "\n";
					if (currentIsFirst) {
						startPos = fastSubnetworkIteration(temp, temp2, tempNodes, tempNodes2,
							target, startPos);
						currentNodeCount = temp2.nodeCount;
						currentIsFirst = false;
						reduction = ((float)temp2.nodeCount) / ((float)temp.nodeCount);
					}
					else {
						startPos = fastSubnetworkIteration(temp2, temp, tempNodes2, tempNodes,
							target, startPos);
						currentNodeCount = temp.nodeCount;
						currentIsFirst = true;
						reduction = ((float)temp.nodeCount) / ((float)temp2.nodeCount);
					}
				}
				if (currentIsFirst) {
					if (currentNodeCount > target) {
						std::cout << "Does not collapse nicely, falling back to overshot Solar System subgraph\n";
						fallbackFlag = true;
						outG = overshotGraph;
						for (int i : overshotNodes) nodes.push_back(i);
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
						std::cout << "Does not collapse nicely, falling back to overshot Solar System subgraph\n";
						fallbackFlag = true;
						outG = overshotGraph;
						for (int i : overshotNodes) nodes.push_back(i);
					}
					else {
						outG = temp2;
						for (int i : tempNodes2) {
							nodes.push_back(i);
						}
					}
				}
				outG.Y = (float*)malloc(sizeof(float) * outG.nodeCount * outG.outputDim);
				lastSubgraphNodeCount = outG.nodeCount;
				std::cout << "Number of subgraph points reached: " << outG.nodeCount << "\n";
				return;
			}

			// Continue Solar System iterations, ping-ponging between temp and temp2.
			// currentIsFirst tracks which buffer holds the latest result.
			bool currentIsFirst = false; // temp2 has the latest result after first iter
			while (true) {
				if (currentIsFirst) {
					// temp has the latest (above-target) graph — iterate into temp2
					solarSystemIteration(temp, temp2, tempNodes, tempNodes2, target);
					std::cout << temp2.nodeCount << "\n";
					if (temp2.nodeCount <= target) {
						// Overshot! Fall back to old method on temp (the previous graph)
						std::cout << "Solar System overshot, falling back to old coarsening on previous-step graph (" << temp.nodeCount << " nodes)\n";
						// Run old fastSubnetworkIteration loop on temp
						Graph<float> fb1(inG.outputDim);
						Graph<float> fb2(inG.outputDim);
						fb1.weighted = inG.weighted;
						fb2.weighted = inG.weighted;
						std::vector<int> fbNodes1 = tempNodes;
						std::vector<int> fbNodes2;
						int startPos = 0;
						startPos = fastSubnetworkIteration(temp, fb2, fbNodes1, fbNodes2,
							target, startPos);
						int cn = fb2.nodeCount;
						bool fbIsFirst = false;
						float fbReduction = 0;
						while (cn > target && (fbReduction < 0.95)) {
							std::cout << cn << "\n";
							if (fbIsFirst) {
								startPos = fastSubnetworkIteration(fb1, fb2, fbNodes1, fbNodes2,
									target, startPos);
								cn = fb2.nodeCount;
								fbIsFirst = false;
								fbReduction = ((float)fb2.nodeCount) / ((float)fb1.nodeCount);
							}
							else {
								startPos = fastSubnetworkIteration(fb2, fb1, fbNodes2, fbNodes1,
									target, startPos);
								cn = fb1.nodeCount;
								fbIsFirst = true;
								fbReduction = ((float)fb1.nodeCount) / ((float)fb2.nodeCount);
							}
						}
						if (fbIsFirst) {
							if (cn > target) {
								std::cout << "Does not collapse nicely, falling back to overshot Solar System subgraph\n";
								fallbackFlag = true;
								outG = temp2;
								for (int i : tempNodes2) nodes.push_back(i);
							}
							else {
								outG = fb1;
								for (int i : fbNodes1) nodes.push_back(i);
							}
						}
						else {
							if (cn > target) {
								std::cout << "Does not collapse nicely, falling back to overshot Solar System subgraph\n";
								fallbackFlag = true;
								outG = temp2;
								for (int i : tempNodes2) nodes.push_back(i);
							}
							else {
								outG = fb2;
								for (int i : fbNodes2) nodes.push_back(i);
							}
						}
						outG.Y =
							(float*)malloc(sizeof(float) * outG.nodeCount * outG.outputDim);
						lastSubgraphNodeCount = outG.nodeCount;
						std::cout << "Number of subgraph points reached: " << outG.nodeCount << "\n";
						return;
					}
					currentIsFirst = false;
				}
				else {
					// temp2 has the latest (above-target) graph — iterate into temp
					solarSystemIteration(temp2, temp, tempNodes2, tempNodes, target);
					std::cout << temp.nodeCount << "\n";
					if (temp.nodeCount <= target) {
						// Overshot! Fall back to old method on temp2 (the previous graph)
						std::cout << "Solar System overshot, falling back to old coarsening on previous-step graph (" << temp2.nodeCount << " nodes)\n";
						Graph<float> fb1(inG.outputDim);
						Graph<float> fb2(inG.outputDim);
						fb1.weighted = inG.weighted;
						fb2.weighted = inG.weighted;
						std::vector<int> fbNodes1 = tempNodes2;
						std::vector<int> fbNodes2;
						int startPos = 0;
						startPos = fastSubnetworkIteration(temp2, fb2, fbNodes1, fbNodes2,
							target, startPos);
						int cn = fb2.nodeCount;
						bool fbIsFirst = false;
						float fbReduction = 0;
						while (cn > target && (fbReduction < 0.95)) {
							std::cout << cn << "\n";
							if (fbIsFirst) {
								startPos = fastSubnetworkIteration(fb1, fb2, fbNodes1, fbNodes2,
									target, startPos);
								cn = fb2.nodeCount;
								fbIsFirst = false;
								fbReduction = ((float)fb2.nodeCount) / ((float)fb1.nodeCount);
							}
							else {
								startPos = fastSubnetworkIteration(fb2, fb1, fbNodes2, fbNodes1,
									target, startPos);
								cn = fb1.nodeCount;
								fbIsFirst = true;
								fbReduction = ((float)fb1.nodeCount) / ((float)fb2.nodeCount);
							}
						}
						if (fbIsFirst) {
							if (cn > target) {
								std::cout << "Does not collapse nicely, falling back to overshot Solar System subgraph\n";
								fallbackFlag = true;
								outG = temp;
								for (int i : tempNodes) nodes.push_back(i);
							}
							else {
								outG = fb1;
								for (int i : fbNodes1) nodes.push_back(i);
							}
						}
						else {
							if (cn > target) {
								std::cout << "Does not collapse nicely, falling back to overshot Solar System subgraph\n";
								fallbackFlag = true;
								outG = temp;
								for (int i : tempNodes) nodes.push_back(i);
							}
							else {
								outG = fb2;
								for (int i : fbNodes2) nodes.push_back(i);
							}
						}
						outG.Y =
							(float*)malloc(sizeof(float) * outG.nodeCount * outG.outputDim);
						lastSubgraphNodeCount = outG.nodeCount;
						std::cout << "Number of subgraph points reached: " << outG.nodeCount << "\n";
						return;
					}
					currentIsFirst = true;
				}
			}
		}


	private:
		template <class T>
		static int fastSubnetworkIteration(Graph<T>& in, Graph<T>& out, std::vector<int>& inNodes, std::vector<int>& outNodes, int target, int startPos = 0) {
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
			for (int i = (startPos + 1) % in.nodeCount; i != startPos; i = (i + 1) % in.nodeCount) {
#endif
				if (partOfNode[i] == -1) {
#ifndef ALLOW_LESS_STRICT_MERGING
					bool neighborsFree = true;
					// Only if none of the nodes that would be included
					// are already part of another cluster
					for (Edge<T>& e : in.edges[i]) {
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
						for (Edge<T>& e : in.edges[i]) {
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
					out.edges[i].push_back(Edge<T>(edge.first, edge.second));
				}
			}

			free(partOfNode);
			free(distanceFromCenter);
			return nextStart;
			};
		template <class T>
		static void getFastSubnetworkOrder(Graph<T>&g, std::vector<std::pair<int, int>>&list) {
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
				};

		static void solarSystemIteration(Graph<float>&in, Graph<float>&out,
			std::vector<int>&inNodes,
			std::vector<int>&outNodes,
			int target) {
			out.edges.clear();
			out.nodeCount = 0;
			outNodes.clear();

			if (in.nodeCount <= 3) {
				out = in;
				outNodes = inNodes;
				return;
			}

			std::vector<int> celestial(in.nodeCount,
				0); // 0=Undef, 1=Sun, 2=Planet, 3=Moon
			std::vector<int> orbitalCenter(in.nodeCount, -1);
			std::vector<float> distanceToOrbit(in.nodeCount, 0.0f);

			std::vector<int> suns;
#ifdef ASSC_SET_ORDER
			std::vector<std::pair<int, int>> candidates(in.nodeCount);
			/*for (int j = 0; j < in.nodeCount; j++)
				candidates[j] = j;*/
			getFastSubnetworkOrder(in, candidates);
#else
			std::vector<int> candidates(in.nodeCount);
			for (int j = 0; j < in.nodeCount; j++)
				candidates[j] = j;
#endif

			static thread_local std::mt19937 rng(std::random_device{}());
			//static thread_local std::mt19937 rng(95837573);//Always use the same seed for performance testing reasons

			// Phase 1 + 2: Select Suns & Planets
#ifdef ASSC_SET_ORDER
			for (int idx = 0; idx < candidates.size(); idx++) {
#else
			while (!candidates.empty()) {
#endif
				std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
#ifdef ASSC_SET_ORDER
				//int idx = candidates.size() - 1;
				int sun = candidates[idx].second;
#else
				int idx = dist(rng);
				int sun = candidates[idx];
				candidates[idx] = candidates.back();
				candidates.pop_back();
#endif
				

				if (celestial[sun] != 0)
					continue;

				bool hasForeignPlanet = false;
				for (Edge<float>& adj : in.edges[sun]) {
					if (celestial[adj.other] != 0) {
						hasForeignPlanet = true;
						break;
					}
				}
				if (hasForeignPlanet)
					continue;

				celestial[sun] = 1;
				suns.push_back(sun);
				for (Edge<float>& adj : in.edges[sun]) {
					celestial[adj.other] = 2;
					orbitalCenter[adj.other] = sun;
					distanceToOrbit[adj.other] = adj.weight;
				}
			}

			// Phase 3: Identify Moons
			for (int v = 0; v < in.nodeCount; v++) {
				if (celestial[v] == 0) {
					celestial[v] = 3;
					std::vector<Edge<float>*> planets;
					for (Edge<float>& adj : in.edges[v]) {
						if (celestial[adj.other] == 2) {
							planets.push_back(&adj);
						}
					}
					if (!planets.empty()) {
						std::uniform_int_distribution<int> dist(0, planets.size() - 1);
						int idx = dist(rng);
						orbitalCenter[v] = planets[idx]->other;
						distanceToOrbit[v] = planets[idx]->weight;
					}
					else {
						celestial[v] = 1;
						suns.push_back(v);
					}
				}
			}

			// Phase 4: Contraction
			out.nodeCount = suns.size();
			out.edges.resize(out.nodeCount);

			std::vector<int> nodeToSunIdx(in.nodeCount, -1);
			for (int j = 0; j < (int)suns.size(); j++) {
				int sun = suns[j];
				outNodes.push_back(inNodes[sun]);
				nodeToSunIdx[sun] = j;
			}

			// Map all non-sun nodes to their sun by walking the orbital chain.
			// This handles promoted nodes correctly: if a planet was promoted to sun,
			// its moons will walk up and find the promoted node as their sun.
			for (int v = 0; v < in.nodeCount; v++) {
				if (celestial[v] == 1)
					continue;
				int curr = v;
				while (curr != -1 && celestial[curr] != 1) {
					curr = orbitalCenter[curr];
				}
				if (curr != -1) {
					nodeToSunIdx[v] = nodeToSunIdx[curr];
				}
			}

			// Compute cumulative distance to assigned sun for edge weight calculation.
			// We use a separate vector to preserve the original hop distances in
			// distanceToOrbit (needed if promotion logic ran above).
			std::vector<float> cumulativeDist(in.nodeCount, 0.0f);
			for (int i = 0; i < in.nodeCount; i++) {
				if (celestial[i] == 1)
					continue;
				float dist = 0;
				int curr = i;
				while (curr != -1 && celestial[curr] > 1) {
					dist += distanceToOrbit[curr];
					curr = orbitalCenter[curr];
				}
				cumulativeDist[i] = dist;
			}

			//std::vector<std::vector<std::pair<int, double>>> newEdges(out.nodeCount);

			for (int j = 0; j < in.nodeCount; j++) {
				int metaI = nodeToSunIdx[j];
				if (metaI == -1)
					continue;
				for (Edge<float>& e : in.edges[j]) {
					int metaOther = nodeToSunIdx[e.other];
					if (metaOther == -1 || metaI >= metaOther)
						continue;

					double dist = cumulativeDist[j] + cumulativeDist[e.other] + e.weight;

					bool contains = false;
					for (auto& ne : out.edges[metaI]) {
						if (ne.other == metaOther) {
							contains = true;
							if (ne.weight > dist) {
								ne.weight = dist;
							}
							break;
						}
					}

					if (!contains) {
						out.edges[metaI].push_back(Edge<float>(metaOther, dist));
					}
				}
			}
			for (int j = out.nodeCount - 1; j >= 0; j--) {
				for (auto& edge : out.edges[j]) {
					out.edges[edge.other].push_back(Edge<float>(j, edge.weight));
				}
			}

			/*for (int j = 0; j < out.nodeCount; j++) {
				for (auto& edge : newEdges[j]) {
					out.edges[j].push_back(Edge<float>(edge.first, edge.second));
				}
			}*/
			out.weighted = true;
		}

		static double firstIterCoarseningRate;
		static int lastSubgraphNodeCount;
		static bool fallbackFlag;
			};
		};