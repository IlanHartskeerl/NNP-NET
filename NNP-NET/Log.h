#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <mutex>

#define LOG_PATH "log.csv"


class Log {
public:
	static void reset(std::string outFile) {
#ifndef LOG_PATH
		return;
#endif
		if (l == nullptr) {
			l = new Log();
		}
		l->filePath = outFile;
		l->header.clear();
		l->rows.clear();
		l->rows.resize(1);
	};

	static void addEntry(int entry, std::string entryName) {
		addEntry(std::to_string(entry), entryName);
	}

	static void addEntry(float entry, std::string entryName) {
		addEntry(std::to_string(entry), entryName);
	}

	static void addEntry(double entry, std::string entryName) {
		addEntry(std::to_string(entry), entryName);
	}

	static void addEntry(std::string entry, std::string entryName) {
		// #ifndef LOG_PATH
		// 			return;
		// #endif
		std::unique_lock<std::mutex> lock(l->m);
		int i = l->rows.size() - 1;
		bool contains = false;
		for (auto& s : l->header) {
			if (s == entryName) {
				contains = true;
				break;
			}
		}
		if (!contains) {
			l->header.push_back(entryName);
		}
		l->rows[i][entryName] = entry;
	};

	static void nextRow() {
#ifndef LOG_PATH
		return;
#endif
		l->rows.push_back(std::unordered_map<std::string, std::string>());
	};

	static void save() {
#ifndef LOG_PATH
		return;
#endif
		std::stringstream ss;
		l->buildOutput(ss);

		std::ofstream file;
		file.open(l->filePath);
		file << ss.str();
		file.close();
	};

private:
	static Log* l;

	Log() {};

	std::mutex m;
	std::string filePath;

	std::vector<std::string> header;

	std::vector<std::unordered_map<std::string, std::string>> rows;

	void buildOutput(std::stringstream& ss) {
		if (header.size() < 1) return;
		ss << header[0];
		for (int i = 1; i < header.size(); i++) {
			ss << ',' << header[i];
		}
		for (auto& row : rows) {
			if (row.size() == 0) continue;
			ss << '\n';
			if (row.count(header[0]) > 0) {
				ss << row[header[0]];
			}
			else {
				ss << "-";
			}
			for (int i = 1; i < header.size(); i++) {
				ss << ",";
				if (row.count(header[i]) > 0) {
					ss << row[header[i]];
				}
				else {
					ss << "-";
				}
			}
		}
	}

};
