#include <vector>
#include <iterator>


/// <summary>
/// I was planing on doing something smart here to get more performance
/// compared to std::unordered_map, but turns out that for this usecase, 
/// just using a default array and linearly iterating over it to check 
/// if a key is in it is just quite a lot faster.
/// The main speedup comes from the deletion stage actually, as 
/// we have a lot of maps, and they all need to delete a number of 
/// individual memory locations. With this implementation, the entire 
/// knn array of "dictionaries" can be freed in a single go, as it is
/// continious in memory.
/// Only downside is that we now have a max on knn, which you would think
/// is not a ploblem for knn, but we are not really doing true knn, more
/// rnn where the r is the lowest number that we have at least k neighbors
/// </summary>
template<class T>
class CustomDict {
public:
	CustomDict() {  };

	bool contains(int i) {
		if (count == 10) {
			return true;
		}
		for (auto p : *this) {
			if (p.first == i) return true;
		}
		return false;
	}

	std::pair<int, T>& at(int i) {
		return data[i];
	}

	void insert(int i, T t) {
		data[count] = { i, t };
		count++;
	}

	std::pair<int, T>* begin() {
		return data;
	}

	std::pair<int, T>* end() {
		return data + count;
	}

	int size() { return count; };

	void clear() { count = 0; };

private:
	std::pair<int, T> data[10];
	int count = 0;
};
