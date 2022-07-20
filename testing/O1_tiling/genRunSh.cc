#include <iostream>
#include <omp.h>
#include <string> 
#include <assert.h>  
#include <vector>
#include <stdio.h>
#include <unordered_set>

using namespace std;

struct tile_factor {
    int Tm, Tn, Tk;
};

int main(int argc, char ** argv){
    int M, N, K;
    int cache_size;
    float factor = 0.5;
    assert(argc >= 5);
    M = stoi(argv[1]);
    N = stoi(argv[2]);
    K = stoi(argv[3]);
    cache_size = stoi(argv[4]);
    if (argc >= 6){
        factor = stof(argv[5]);
    }
    
    std::vector<tile_factor> tile_factors;

    int max_mnk = max(max(M, N), K);
    #pragma omp parallel
    {
        std::vector<tile_factor> private_tile_factors;
        #pragma omp for
        for (int Tm = 1; Tm <= max_mnk; Tm++){
            for (int Tn = Tm; Tn <= max_mnk; Tn ++ ){
                for (int Tk = Tn; Tk <= max_mnk; Tk++){
                    int mem_used = Tm * Tn + Tn * Tk + Tm * Tk;
                    if ((mem_used <= cache_size) && (mem_used >= cache_size * factor)){
                        int candidate[6][3] = {{Tm, Tn, Tk}, {Tm, Tk, Tn}, {Tn, Tm, Tk}, {Tn, Tk, Tm}, {Tk, Tn, Tm}, {Tk, Tm, Tn}};
                        for (int i = 0; i < 6; i++){
                            tile_factor t;
                            t.Tm = candidate[i][0];
                            t.Tn = candidate[i][1];
                            t.Tk = candidate[i][2];
                            bool duplicate = false;
                            for (int j = 0; j < i; j++){
                                if (candidate[j][0] == candidate[i][0] && candidate[j][1] == candidate[i][1] && candidate[j][2] == candidate[i][2]){
                                    duplicate = true;
                                }
                            }
                            if (!duplicate && t.Tm <= M && t.Tn <= N && t.Tk <= K){
                                private_tile_factors.push_back(t);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp critical 
        {
            for (auto t: private_tile_factors)
                tile_factors.push_back(t);
        }
    }
    std::unordered_set<int> visited;
    for (auto i = tile_factors.begin(); i!=tile_factors.end();){
        int key = i->Tm * N * K + i->Tn * K + i->Tk;
        assert(i->Tm <= M && i->Tn <= N && i->Tk <= K);
        int memused = (i->Tm * i->Tn + i->Tn * i->Tk + i->Tm * i->Tk);
        assert(memused <= cache_size);
        assert(memused >= (cache_size * factor));
        if (visited.count(key))
            i = tile_factors.erase(i);
        else {
            visited.insert(key);
            i++;
        }
    }
    printf("%ld candidates in all.\n", tile_factors.size());
    return 0;
}