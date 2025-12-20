// sim.cpp
// Single-file C++ port of the Python code you supplied.
// Compile: g++ -O3 -std=c++17 DEBUGsimMultiThreadXoroshiroTilingVector.cpp -o DEBUSsimMultiThreadXoroshiroTilingVector -pthread
// Run: ./DEBUSsimMultiThreadXoroshiroTiling 1 5 0 3.3 512 2 2 1 20 60 0 0.01 0.2 0.01 1321  8076 0.1 5000
//
// Outputs CSV files:
//  - scoreSnaps.csv (snapshots of cumulative score at snapshot times)
//  - totalScore.csv (final total scores grid)
//  - ruleSnaps.csv (flattened rule snapshots: snap, y, x, ruleIndex, value)
//  - nonCumulativeScore.csv (snapshots of scoreTracker per snap)
//


/*run a version of this wim without the experiment py file using 
./sim p00 p01 p10 p11 gridN res0 res1 maxN rounds iters snaps evolutionRate mutationRate evolutionChance seed1 seed2 inversionpercent inversion round
./sim 1 5 0 3 64 4 4 1 10000 60 100 0.01 0.001 0.2 3 2 0 5000

*/

//This version I make it so that assigning matchups doesn't use sin and cos. 
#define NOMINMAX
#include <fstream>
#include <thread>
#include <random>
#include <cmath>       // if you use math functions
#include <chrono>      // if you use std::chrono for timing
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <queue>
#include <stack>
#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <memory>
#include <array>
#include "RandomGenerator/Xoshiro.hpp" //random number generator header library from https://www.pcg-random.org/download.html 
using namespace std;

/* ---------------------------
   Utility / random helpers
   --------------------------- */

using u64 = unsigned long long;
XoshiroCpp::Xoroshiro128Plus grid_rng;
XoshiroCpp::Xoroshiro128Plus global_rng;
std::chrono::high_resolution_clock::time_point setUpEndTime; //note: looked on stack overflow for type because documentation was confusing https://stackoverflow.com/questions/31497531/what-is-the-type-of-stdchronohigh-resolution-clocknow-in-c11#:~:text=Okay%2C%20I%20see%20the%20error,1 


double uniform01() {
    return std::uniform_real_distribution<double>(0.0, 1.0)(global_rng);
}

int randint(int a, int b) { // inclusive [a,b]
    return std::uniform_int_distribution<int>(a,b)(global_rng);
}

/* ---------------------------
   Perlin / Fractal noise
   Ported from your perlin_numpy implementation
   --------------------------- */

static inline double interpolant(double t) {
    // t * t * t * (t * (t * 6 - 15) + 10)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Helper: create 3D array-like vectors: shape: (n, h, w)
using Noise3D = vector<vector<vector<double>>>;

// generate_perlin_noise_2d(shape, res, tileable, seed, number) 
Noise3D generate_perlin_noise_2d(pair<int,int> shape, pair<int,int> res, pair<bool,bool> tileable, unsigned seed, int number=1) {
    int H = shape.first;
    int W = shape.second;
    Noise3D data(number, vector<vector<double>>(H, vector<double>(W, 0.0)));

    for (int n=0;n<number;++n) {
        double delta0 = (double)res.first / (double)H;
        double delta1 = (double)res.second / (double)W;
        int d0 = H / res.first;
        int d1 = W / res.second;

        // grid coords
        // grid[i][j] = (u, v) where u and v in [0,1)
        vector<vector<pair<double,double>>> grid(H, vector<pair<double,double>>(W));
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                double gx = fmod((i * delta0), 1.0);
                double gy = fmod((j * delta1), 1.0);
                grid[i][j] = {gx, gy};
            }
        }

        // gradients
        std::uniform_real_distribution<double> angDist(0.0, 2.0 * M_PI);
        vector<vector<pair<double,double>>> gradients(res.first+1, vector<pair<double,double>>(res.second+1));
        for (int i=0;i<=res.first;++i) for (int j=0;j<=res.second;++j) {
            double ang = angDist(grid_rng);
            gradients[i][j] = {cos(ang), sin(ang)};
        }
        if (tileable.first) for (int j=0;j<=res.second;++j) gradients[res.first][j] = gradients[0][j];
        if (tileable.second) for (int i=0;i<=res.first;++i) gradients[i][res.second] = gradients[i][0];

        // expand gradients to pixel resolution by repeating blocks
        auto gradAt = [&](int i, int j)->pair<double,double> {
            // which cell in gradient grid corresponds?
            int gi = min(i / d0, res.first); // careful with edge
            int gj = min(j / d1, res.second);
            return gradients[gi][gj];
        };

        // compute g00,g10,g01,g11 and ramps
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                // grid fractional coords
                double gx = grid[i][j].first;
                double gy = grid[i][j].second;

                // indices for gradients (top-left corner of cell)
                int cell_i = (int)floor((double)i / d0);
                int cell_j = (int)floor((double)j / d1);
                // clamp to grid
                cell_i = std::min(cell_i, res.first);
                cell_j = std::min(cell_j, res.second);

                // get gradients for corners
                auto g00 = gradients[cell_i    ][cell_j    ];
                auto g10 = gradients[min(cell_i+1,res.first)][cell_j    ];
                auto g01 = gradients[cell_i    ][min(cell_j+1,res.second)];
                auto g11 = gradients[min(cell_i+1,res.first)][min(cell_j+1,res.second)];

                // compute dot products
                double n00 = g00.first * gx       + g00.second * gy;
                double n10 = g10.first * (gx-1.0) + g10.second * gy;
                double n01 = g01.first * gx       + g01.second * (gy-1.0);
                double n11 = g11.first * (gx-1.0) + g11.second * (gy-1.0);

                // interpolation
                double tx = interpolant(gx);
                double ty = interpolant(gy);
                double n0 = n00*(1.0 - tx) + tx * n10;
                double n1 = n01*(1.0 - tx) + tx * n11;
                double val = sqrt(2.0) * ((1.0 - ty) * n0 + ty * n1);
                data[n][i][j] = val; //n is different layers of noise maps (there are five). the first four each give probability of cooperating depending on the last round (CC,CD,DC, or DD). what the last round was determines which map it accesses. 
            }
        }
    }
    return data;
}

Noise3D generate_fractal_noise_2d(pair<int,int> shape, pair<int,int> res, int octaves=1, double persistence=0.5, double lacunarity=2.0, pair<bool,bool> tileable={false,false}, unsigned seed=0, int number=1) {
    Noise3D noise(number, vector<vector<double>>(shape.first, vector<double>(shape.second, 0.0)));
    // generate base perlin at base frequency, then add octaves
    for (int n=0;n<number;++n) {
        double frequency = 1.0;
        double amplitude = 1.0;
        // we can't re-use generate_perlin_noise_2d for differing frequencies easily,
        // so call it with appropriate res for each octave and add scaled results.
        for (int o=0;o<octaves;++o) {
            pair<int,int> r = { int(frequency * res.first), int(frequency * res.second) };
            auto base = generate_perlin_noise_2d(shape, r, tileable, seed + n + o*97, 1);
            for (int i=0;i<shape.first;++i) for (int j=0;j<shape.second;++j) {
                noise[n][i][j] += amplitude * base[0][i][j];
            }
            frequency *= lacunarity;
            amplitude *= persistence;
        }
    }
    return noise;
}

/* ---------------------------
   MemoryN classes (agents)
   --------------------------- */

enum Move: int {
    COOP = 1,
    DEF = 0
};

struct Memory1 {
    Move startMove;
    Move prevMove;
    array<double,4> rule;
    double score;
    string name;
    double mutationRate;

    Memory1(Move _start = COOP, double _mutationRate=0.0):
        startMove(_start),
        prevMove(startMove),
        rule{0.0,0.0,0.0,0.0},
        mutationRate(_mutationRate),
        score(0.0),
        name("MemoryN")
    {}

    virtual void startup(Move _start) {
        this->startMove = _start;
        this->prevMove = _start;
        this->score = 0.0;
    }

    int playMove(int theirPrev, double seed, int roundNum) { //I wonder what the branch mispredictions are like
        // if (roundNum == 0) {
        //     prevMove = startMove;
        //     return startMove;
        // } Omit in favor of doing this check when running the individual game, rather than at every call to playMove
        int key = (prevMove << 1) | theirPrev;
        double prob = rule[key];
        if (seed < prob) {
            prevMove = COOP;
            return COOP;
        } else {
            prevMove = DEF;
            return DEF;
        }
    }

    void reset() {
        score = 0.0;
        prevMove = startMove;
    }

    virtual string repr() const {
        return name;
    }

    void setRule(const array<double, 4> &r) {
        rule[0] = r[0];
        rule[1] = r[1];
        rule[2] = r[2];
        rule[3] = r[3];
    }
};

struct BLANK : public Memory1 {
    BLANK(Move start = COOP) : Memory1(start) {
        name = "BLANK";
        rule[0] = 0.0;
        rule[1] = 0.0;
        rule[2] = 0.0;
        rule[3] = 0.0;
        startMove = start;
        prevMove = start;
    }
};

/* ---------------------------
Grid generation
--------------------------- */

using AgentGrid = vector<vector<shared_ptr<Memory1>>>;

// blankGrid(N, res, maxN=1, seed)
AgentGrid blankGrid(int N, pair<int,int> res, unsigned seed = 0, double mutationRate=0.0) {
    if (seed==0) seed = (unsigned) (uniform01() * 1000.0);
    //cout << "seed: " << seed << "\n";
    constexpr int number = 5; // 4 rules, 1 mutation rate - huh??
    auto paramMaps = generate_fractal_noise_2d(
        {N,N}, 
        res, 
        /*octaves=*/1, 
        /*persistence=*/0.5, 
        /*lacunarity=*/2.0, 
        {false,false}, 
        seed, 
        number);
    // paramMaps is number x N x N
    // python code adds +1.3 then divides by 2
    for (int k=0;k<number;++k) for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        paramMaps[k][i][j] += 1;
        paramMaps[k][i][j] /= 2.0;
    }
    AgentGrid grid(N, vector<shared_ptr<Memory1>>(N)); //would it be a good idea to dynamically allocate this so that it is not copying a massive datastructure?
    for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        auto ag = make_shared<BLANK>(COOP);
        // create rule vector of length 4. The python did: setRule([i**2 for i in list(paramMaps[:,idr,idc])])
        // paramMaps[:,idr,idc] is "number" values — they square them.
        array<double,4> ruleVals;
        // If number (channels) does not equal rule size, we'll distribute or repeat
        // but the Python used list(paramMaps[:,idr,idc]) and squared those values to form rule,
        // meaning rule length == number. But rule length should be 4**maxN. In the python blankGrid,
        // they use number=(4**maxN) so number == rule length. So here it's consistent.
        for (int k=0;k<number-1;++k) {
            double v = paramMaps[k][i][j];
            ruleVals[k] = v;
            // clamp [0,1]
            if (ruleVals[k] < 0.0) ruleVals[k] = 0.0;
            if (ruleVals[k] > 1.0) ruleVals[k] = 1.0;
        }
        ag->setRule(ruleVals);
        ag->mutationRate = mutationRate;//paramMaps[4][i][j];
        grid[i][j] = ag;
    }
    return grid;
}

/* ---------------------------
Tournaments
--------------------------- */

static vector<vector<double>> payoffMatrix = {{1,5},{0,3}};

struct TorusResult {
    // snapshots: vector of 2D arrays (snap index -> N x N)
    vector<vector<vector<double>>> scoreSnaps; // snap -> y -> x 
    vector<vector<vector<double>>> ruleSnaps;  // snap -> y -> x*ruleLen (flattened per x) QUESTION - confused here? what does flattened mean?
    vector<vector<vector<double>>> nonCumulativeScoreSnaps; // snap -> y -> x
    vector<vector<double>> totalScore; // y x
};

int flatten_index(int y, int x, int X) { return y*X + x; }

vector<vector<pair<int,int>>> pickOpponents(const AgentGrid &agentGrid) {
    int yLen = (int)agentGrid.size();
    int xLen = (int)agentGrid[0].size();
    int N = yLen * xLen; //PA note: N never changes does it? why not make it a field of agent grid? (or better yet, if you can get parameters at compile time, calculate it at compile time)
    vector<double> angles(N);
    for (int i=0;i<N;++i) angles[i] = uniform01() * 2.0 * M_PI; //list of randomly generated angles. Questions: do you really need to make this array here? why not calculate an angle and then put it in xs and xy directly
    vector<int> xs(N), ys(N);
    for (int i=0;i<N;++i) {
        xs[i] = (int)round(cos(angles[i]));
        ys[i] = (int)round(sin(angles[i])); //<xs[i],ys[i]> is a unit vector in direction angle[i]
    }
    vector<vector<pair<int,int>>> opponent(yLen, vector<pair<int,int>>(xLen));
    for (int iy=0; iy<yLen; ++iy) {
        for (int ix=0; ix<xLen; ++ix) {
            int id = iy * xLen + ix; //id maps an index in the 2d array to an index in angles, xs, and xy
            int xLoc = (ix + xs[id]) % xLen; 
            if (xLoc < 0) xLoc += xLen;
            int yLoc = (iy + ys[id]) % yLen;
            if (yLoc < 0) yLoc += yLen;
            opponent[iy][ix] = {xLoc, yLoc}; //why do all the angle stuff? why not just pick an element neighboring the cell with a certain probability (if you wanted you could calculate the probability of corner vs staight on)
        }
    }
    return opponent;
}

static const pair<int,int> DIRS[8] = {
    { 1, 0}, {-1, 0}, {0, 1}, {0,-1},
    { 1, 1}, { 1,-1}, {-1, 1}, {-1,-1}
};

vector<vector<pair<int,int>>> pickOpponentsNew(const AgentGrid &agents) {
    int Y = agents.size();
    int X = agents[0].size();

    vector<vector<pair<int,int>>> opp(Y, vector<pair<int,int>>(X));

    for(int y=0; y<Y; ++y) {
        for(int x=0; x<X; ++x) {
            int d = rand() % 8;
            int nx = (x + DIRS[d].first + X) % X;
            int ny = (y + DIRS[d].second + Y) % Y;
            opp[y][x] = {nx, ny};
        }
    }
    return opp;
}

vector<vector<array<double,5>>> agentRuleSnapshot(const AgentGrid &agents) {
    int Y = agents.size();
    int X = agents[0].size();

    vector<vector<array<double,5>>> snap(Y, vector<array<double,5>>(X));
    for(int i=0; i<Y; ++i){
        for(int j=0; j<X; ++j) {
            for(int k=0; k<4; ++k) snap[i][j][k] = agents[i][j]->rule[k];
            snap[i][j][4] = agents[i][j]->mutationRate;
        }
    }
    return snap;
}
void printPlayed(vector<vector<int>> played){
    cout<<"printing played 2D array"<<endl;
    for (int i=0;i<played.size();i++){
        for(int j=0;j<played.size();j++){
            cout<<played[i][j]<<", ";
        }
        cout<<endl;
    }
}
void wrapRow(int tileRowIndex, int globalRowIndex, int tileStartX, int lastStart, int gxi, int txi, int endxi, int gridN, vector<vector<double>>& globalScoreTracker, vector<vector<int>>& globalPlayedTracker, vector<vector<double>>& threadScoreTracker, vector<vector<int>>& threadPlayedTracker){
    if (tileStartX==0){ //top left corner of thread accumulator when it needs to wrap
        globalScoreTracker[globalRowIndex][gridN-1]+=threadScoreTracker[tileRowIndex][txi];
        threadScoreTracker[tileRowIndex][txi]=0;
        globalPlayedTracker[globalRowIndex][gridN-1]+=threadPlayedTracker[tileRowIndex][txi];
        threadPlayedTracker[tileRowIndex][txi]=0;
        gxi+=1; //taking the left columb off of future loops because it needs to be wrapped
        txi+=1;
    }
    if (tileStartX==lastStart){ //top right corner of thread accumulator when it needs to wrap
        globalScoreTracker[globalRowIndex][0]+=threadScoreTracker[tileRowIndex][endxi-1]; //ednxi is the size of a dimension thread accumulator vector. it is decremented once i have wrapped because I am ignoring it. 
        threadScoreTracker[tileRowIndex][endxi-1]=0;
        globalPlayedTracker[globalRowIndex][0]+=threadPlayedTracker[tileRowIndex][endxi-1]; //ednxi is the size of a dimension thread accumulator vector. it is decremented once i have wrapped because I am ignoring it. 
        threadPlayedTracker[tileRowIndex][endxi-1]=0;
        endxi-=1;
    }
    for (int gx = gxi, tx=txi; tx<endxi;tx++,gx++){ //this doesn't touch the first or last column in case they need to be wrapped
        globalScoreTracker[globalRowIndex][gx]+=threadScoreTracker[tileRowIndex][tx];
        threadScoreTracker[tileRowIndex][tx]=0;
    }
    for (int gx = gxi, tx=txi; tx<endxi;tx++,gx++){ //this doesn't touch the first or last column in case they need to be wrapped
        globalPlayedTracker[globalRowIndex][gx]+=threadPlayedTracker[tileRowIndex][tx];
        threadPlayedTracker[tileRowIndex][tx]=0;
    }
}

void accumulate(vector<thread>& threads, vector<array<int,2>>& tileStartCoords, vector<vector<vector<double>>>& scoreTracker_threads,vector<vector<vector<int>>>& playedTracker_threads, vector<vector<double>>& scoreTracker,vector<vector<int>>& playedTracker,int gridN,int lastStart){
    int numthreads = threads.size();
    for (auto &th : threads){
        if (th.joinable()){
            th.join();
        } 
    } 
    threads.clear();

    // cout<<"accumulate called numThreads is "<<numthreads<<" and starting global played tracker is ";
    // printPlayed(playedTracker);
    // cout<<endl<<endl;


    for (int tid=0;tid<numthreads;tid++){
        int tileStartY = tileStartCoords[tid][0];
        int tileStartX = tileStartCoords[tid][1];

        // cout<<"thread "<<tid<<" has start cord ("<<tileStartY<<","<<tileStartX<<"). its thread specific played tracker is below ";
        // printPlayed(playedTracker_threads[tid]);
        // cout<<endl;
        
        int gyi=tileStartY-1;  //these are the initial values to use in the loops below. //gy is the coordinate on the global accumulator grid, and ty is the cooresponding coordinate on the thread specific accumulator grid
        int gxi=tileStartX-1;
        int tyi = 0;
        int txi=0;
        int endyi=scoreTracker_threads[0].size();
        int endxi=scoreTracker_threads[0].size();

        //this is the math for the edge of the torus where a threads accumulator vector wraps off of the end of the global accumulator vector
        if (tileStartY==0 ){ //top row 
            wrapRow(0, gridN-1, tileStartX, lastStart, gxi, txi, endxi, gridN, scoreTracker, playedTracker, scoreTracker_threads[tid], playedTracker_threads[tid]);

            wrapRow(0, gridN-1, tileStartX, lastStart, gxi, txi, endxi, gridN, scoreTracker, playedTracker, scoreTracker_threads[tid], playedTracker_threads[tid]);
            tyi++;
            gyi++;
        }
        if (tileStartY==lastStart){
            wrapRow(endyi-1, 0, tileStartX, lastStart, gxi, txi, endxi, gridN, scoreTracker, playedTracker, scoreTracker_threads[tid], playedTracker_threads[tid]);
            endyi--;
        }
        //by the time it gets here, anything that needed to be wrapped in both directions has already been wrapped. tyi, gyi, and edyi give the rows i need to deal with
        //this loop has bad locality, but the alternative is to put an if condition in a way bigger loop, which would be more expensive i think 
        if (tileStartX == 0){ //left col
            for (int gy = gyi, ty=tyi; ty<endyi;ty++,gy++){ //this doesn't touch the first or last column in case they need to be wrapped. 
                scoreTracker[gy][gridN-1]+=scoreTracker_threads[tid][ty][0];
                scoreTracker_threads[tid][ty][0]=0;
            }
            //seperate loops for better locality
            for (int gy = gyi, ty=tyi; ty<endyi;ty++,gy++){ //this doesn't touch the first or last column in case they need to be wrapped. 
                playedTracker[gy][gridN-1]+=playedTracker_threads[tid][ty][0];
                playedTracker_threads[tid][ty][0]=0;
            }
            txi++;
            gxi++;
        }
        if (tileStartX==lastStart){ //right col
            for (int gy = gyi, ty=tyi; ty<endyi;ty++,gy++){ //this doesn't touch the first or last column in case they need to be wrapped. 
                scoreTracker[gy][0]+=scoreTracker_threads[tid][ty][endxi-1];
                scoreTracker_threads[tid][ty][endxi-1]=0;
            }
            for (int gy = gyi, ty=tyi; ty<endyi;ty++,gy++){ //this doesn't touch the first or last column in case they need to be wrapped. 
                playedTracker[gy][0]+=playedTracker_threads[tid][ty][endxi-1];
                playedTracker_threads[tid][ty][endxi-1]=0;
            }
            endxi--;
        }

            
        //This handles the interior of tile and any edges that are not also edges of the agent grid 
        for (int gy = gyi,ty =tyi; ty<endyi; ty++,gy++){ //gy is the coordinate on the global accumulator grid, and ty is the cooresponding coordinate on the thread specific accumulator grid
            for(int gx=gxi,tx=txi; tx<endxi; tx++,gx++){
                scoreTracker[gy][gx]+=scoreTracker_threads[tid][ty][tx];
                scoreTracker_threads[tid][ty][tx]=0;
            }
        }

        for (int gy = gyi,ty =tyi; ty<endyi;gy++,ty++){ //gy is the coordinate on the global accumulator grid, and ty is the cooresponding coordinate on the thread specific accumulator grid
            for(int gx=gxi,tx=txi; tx<endxi;gx++,tx++){
                playedTracker[gy][gx]+=playedTracker_threads[tid][ty][tx];
                playedTracker_threads[tid][ty][tx]=0;
            }
        }
    }
    // cout<<endl<<"after summing all the played trackers. the global played tracker is as follows"<<endl;
    // printPlayed(playedTracker);
    // cout<<endl<<endl<<endl<<endl;


}





TorusResult torusTournament(AgentGrid agentGrid, int iters, int rounds, int snaps, float evolutionRate, //could agentGrid be passed by reference?
    float evolutionChance, float mutationRate, float inversionPercentage, int inversionRound, int tileSize) {

    int yLen = (int)agentGrid.size();
    int xLen = (int)agentGrid[0].size();
    int N = yLen * xLen;
    TorusResult out;
    vector<vector<double>> totalScore(yLen, vector<double>(xLen, 0.0));
    int snapEvery;
    if (snaps==0){
        snapEvery=rounds+1;
    }
    else{
        snapEvery = max(1, rounds / snaps);
    }
    vector<vector<double>> paddedScore(yLen+2, vector<double>(xLen+2, 0.0));

    setUpEndTime = chrono::high_resolution_clock::now();

    for (int round=0; round<rounds; ++round) {
        if (round == inversionRound) {
            //invertCentralBlock(agentGrid, inversionPercentage);
        }
        // PLAY MATCHES
        auto matchups = pickOpponents(agentGrid);
        vector<vector<int>> playedTracker(yLen, vector<int>(xLen, 0));
        vector<vector<double>> scoreTracker(yLen, vector<double>(xLen, 0.0));

        // BEFORE launching threads: create deterministic thread seeds and decide nThreads
        int nThreads = std::min(static_cast<int>(std::thread::hardware_concurrency()), (int) yLen);
        if (nThreads < 1) nThreads = 1;
        //nThreads=1;

        // // Create thread seeds deterministically using global_rng (seeded in main)
        // vector<uint64_t> thread_seeds(nThreads);
        // for (int t = 0; t < nThreads; ++t) {
        //     thread_seeds[t] = global_rng(); // deterministic sequence
        // }

        // Prepare per-thread accumulators

        vector<vector<vector<double>>> scoreTracker_threads(nThreads, //initialize this with the correct number. might be num threads or tile size?
            vector<vector<double>>(tileSize+2, vector<double>(tileSize+2, 0.0)));
        vector<vector<vector<int>>> playedTracker_threads(nThreads,
            vector<vector<int>>(tileSize+2, vector<int>(tileSize+2, 0))); //each tile is dependent on the neighbors in either direction. (1,1) in the accumulator is the coordinate of (startY, startX) 

        // Worker now receives thread id and seed; 
        auto worker = [&](int t_id, int rngSeed, int startY, int startX, int endY, int endX) {
            XoshiroCpp::Xoroshiro128Plus local_rng(rngSeed); //Question: do you really need 64 bits of randomness? and/or could a thread use smaller parts of a random number before generating a new one. 
            std::uniform_real_distribution<double> unif(0.0, 1.0);

            auto local_uniform01 = [&](){ return unif(local_rng); };

            // local references to thread-local accumulators
            auto &scoreTracker_local = scoreTracker_threads[t_id];
            auto &playedTracker_local = playedTracker_threads[t_id];

            for (int idy = startY; idy < endY; ++idy) {
                for (int idx = startX; idx < endX; ++idx) { //endY is exclusive
                    
                    pair<int,int> match = matchups[idy][idx];
                    auto a1 = agentGrid[idy][idx];
                    auto a2 = agentGrid[match.second][match.first];

                    int localMatchCoordY = match.second;
                    if (localMatchCoordY<startY){
                        localMatchCoordY=endY;
                    }
                    else if(localMatchCoordY>endY){
                        localMatchCoordY=startY-1;
                    }
                    int localMatchCoordX = match.first;
                    if (localMatchCoordX<startX){
                        localMatchCoordX=endX;
                    }
                    else if(localMatchCoordX>endX){
                        localMatchCoordX=startX-1;
                    }

                    // increment played count for both players in THREAD-LOCAL arrays
                    
                    playedTracker_local[idy-startY+1][idx-startX+1] += 1; 
                    playedTracker_local[localMatchCoordY-startY+1][localMatchCoordX-startX+1] += 1; //TODO THE MATCHUPS WRAP!! I have to unwrap them - oh wait, I can must make matchups return the unwrapped version too. I should probably have two version of the worker. One to be called on tiles with no edges to save work

                    // generate seeds for iterated plays using local_rng
                    vector<double> seeds(2 * iters);
                    for (int s = 0; s < 2*iters; ++s) seeds[s] = local_uniform01();

                    


                    for (int n = 0; n < iters; ++n) {
                        unsigned long long a1prev = a1->prevMove;
                        unsigned long long a2prev = a2->prevMove;
                        int a1move = a1->playMove(a2prev, seeds[n], n);
                        int a2move = a2->playMove(a1prev, seeds[n+iters], n);
                        // accumulate into thread-local arrays
                        scoreTracker_local[idy-startY+1][idx-startX+1] += payoffMatrix[a1move][a2move];
                        scoreTracker_local[localMatchCoordY-startY+1][localMatchCoordX-startX+1] += payoffMatrix[a2move][a1move];
                    }
                    a1->reset();
                    a2->reset();
                }
            }
        };

        
       
        vector<thread> threads;
        vector<array<int,2>> tileStartCoords(nThreads);
        int threadid=0;
        int currentThreads=0;
        int numThreadsMadeSoFar=0;
        //std::cout<<"nThreads is "<<nThreads<<endl;
        int lastStart = agentGrid.size()-tileSize;
        for (int startY=0;startY<=lastStart;startY+=tileSize){
            for (int startX = 0; startX<=lastStart;startX+=tileSize) {
                // std::cout<<"about to make thread num "<<numThreadsMadeSoFar<<endl;
                 threads.emplace_back(worker,threadid,global_rng(),startY,startX,startY+tileSize,startX+tileSize);
                //  std::cout<<"made thread num "<<numThreadsMadeSoFar<<endl;
                 numThreadsMadeSoFar++;
                tileStartCoords[currentThreads][0]=startY;                     //tile startCoords hold the coordinate int the global grid that (1,1) in the threads accumulator vector will map to
                tileStartCoords[currentThreads][1]=startX;
                currentThreads++;
                threadid++;
                if (currentThreads==nThreads){
                    accumulate(threads, tileStartCoords, scoreTracker_threads, playedTracker_threads, scoreTracker, playedTracker,xLen,lastStart);
                    threadid=0;
                    currentThreads=0;
                }
            }
        }
        if (threads.size()>0){ 
            accumulate(threads, tileStartCoords, scoreTracker_threads, playedTracker_threads, scoreTracker, playedTracker,xLen,lastStart);
        }
        
        

        

        // for (int t=0; t<nThreads; ++t) {
        //     for (int i=0;i<yLen;++i) {
        //         for (int j=0;j<xLen;++j) {
        //             playedTracker[i][j] += playedTracker_threads[t][i][j];
        //             scoreTracker[i][j] += scoreTracker_threads[t][i][j];
        //         }
        //     }
        // }

        // Normalize by playedTracker (avoid div by zero)
        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) {
            if (playedTracker[i][j] > 0) scoreTracker[i][j] /= (double)playedTracker[i][j];
            totalScore[i][j] += scoreTracker[i][j];
        }

        // Evolution
        AgentGrid newGrid = agentGrid; // shallow copy of shared_ptrs
        double shiftPercentage = 0.2;
        double mutationRate = 0.01;
        // build padded score toroidally
        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) paddedScore[i+1][j+1] = scoreTracker[i][j];
        // wrap edges
        for (int j=0;j<xLen;++j) paddedScore[0][j+1] = scoreTracker[yLen-1][j];
        for (int j=0;j<xLen;++j) paddedScore[yLen+1][j+1] = scoreTracker[0][j];
        for (int i=0;i<yLen;++i) paddedScore[i+1][0] = scoreTracker[i][xLen-1];
        for (int i=0;i<yLen;++i) paddedScore[i+1][xLen+1] = scoreTracker[i][0];
        paddedScore[0][0] = scoreTracker[yLen-1][xLen-1];
        paddedScore[yLen+1][xLen+1] = scoreTracker[0][0];
        paddedScore[0][xLen+1] = scoreTracker[yLen-1][0];
        paddedScore[yLen+1][0] = scoreTracker[0][xLen-1];

        // decide evolution for each agent
        vector<double> evolveVec(N);
        for (int i=0;i<N;++i) evolveVec[i] = uniform01();
        double chance = 0.1;
        int count = -1;
        for (int idy=0; idy<yLen; ++idy) {
            for (int idx=0; idx<xLen; ++idx) {
                ++count;
                // find max in local 3x3 window
                int yMin = idy;
                int yMax = idy+3;
                int xMin = idx;
                int xMax = idx+3;
                // padded region is paddedScore[yMin:yMax, xMin:xMax] (size 3x3)
                double mx = -1e300;
                int bestIndex = 0;
                for (int py=yMin; py<yMax; ++py) for (int px=xMin; px<xMax; ++px) {
                    double val = paddedScore[py][px];
                    int linear = (py - yMin) * 3 + (px - xMin);
                    if (val > mx) { mx = val; bestIndex = linear; }
                }
                // map bestIndex to neighbor coords (unwrapped)
                int uy = ((bestIndex / 3) - 1) + idy;
                int ux = ((bestIndex % 3) - 1) + idx;
                // wrap
                if (uy < 0) uy = yLen + uy;
                else if (uy >= yLen) uy -= 1;
                if (ux < 0) ux = xLen + ux;
                else if (ux >= xLen) ux -= 1;

                // shift
                vector<double> ruleShift(agentGrid[idy][idx]->rule.size(), 0.0);
                if (evolveVec[count] < chance) {
                    auto &src = agentGrid[uy][ux]->rule;
                    auto &dst = agentGrid[idy][idx]->rule;
                    for (size_t k=0;k<dst.size();++k) ruleShift[k] = (src[k] - dst[k]) * shiftPercentage;
                }
                // mutate
                vector<double> ruleShift2(agentGrid[idy][idx]->rule.size(), 0.0);
                for (size_t k=0;k<ruleShift2.size();++k) {
                    ruleShift2[k] = ((uniform01() * 2.0) - 1.0) * mutationRate;
                }
                array<double,4> newRule = agentGrid[idy][idx]->rule;
                for (size_t k=0;k<newRule.size();++k) newRule[k] = newRule[k] + ruleShift[k] + ruleShift2[k];
                for (size_t k=0;k<newRule.size();++k) {
                    if (newRule[k] < 0.0) newRule[k] = 0.0;
                    if (newRule[k] > 1.0) newRule[k] = 1.0;
                }
                // assign to newGrid copy
                // make a fresh BLANK agent to hold new rule while preserving other meta
                auto newAgent = make_shared<BLANK>(agentGrid[idy][idx]->startMove);
                newAgent->name = agentGrid[idy][idx]->name;
                newAgent->rule = newRule;
                newAgent->startMove = agentGrid[idy][idx]->startMove;
                newAgent -> mutationRate = agentGrid[idy][idx]->mutationRate;
                newAgent->prevMove = agentGrid[idy][idx]->prevMove;
                newGrid[idy][idx] = newAgent;
            }
        }
        agentGrid = newGrid;

        if ((round == 1 && snaps!=0) || ((round % snapEvery) == 0 && (round / snapEvery) > 0)) {
            // push snapshots
            out.scoreSnaps.push_back(totalScore);
            auto ruleSnap = agentRuleSnapshot(agentGrid);
            // For storage simplicity, push ruleSnaps as flattened vectors per cell
            // but we'll convert to vector<vector<vector<double>>> where innermost is concatenated rule vector per cell
            size_t ruleLen = 5;//(int)agentGrid[0][0]->rule.size();
            // flatten rules into 2D matrix of (y, x*ruleLen) to mimic original
            vector<vector<double>> flatRules(yLen, vector<double>(xLen * ruleLen));
            for (int iy=0; iy<yLen; ++iy) for (int ix=0; ix<xLen; ++ix) {
                for (int k=0;k<ruleLen;++k) flatRules[iy][ix*ruleLen + k] = ruleSnap[iy][ix][k];
            }
            // store flatRules into ruleSnaps (but as 3D: snap -> y -> x*ruleLen)
            out.ruleSnaps.push_back(flatRules);
            out.nonCumulativeScoreSnaps.push_back(scoreTracker);
            //cout << "progress: " << (round / snapEvery) << " / "<<snaps<<"\n";
        }
        // zero out global trackers 
        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) {
            playedTracker[i][j] = 0;
            scoreTracker[i][j] = 0.0;
        }
        
    } // end rounds

    out.totalScore = totalScore;
    return out;
}

/* ---------------------------
Output helpers (CSV)
--------------------------- */

void write_scoreSnaps_csv(const vector<vector<vector<double>>> &scoreSnaps, const string &fname) {
    ofstream f(fname);
    // write each snap as flattened row; comment header
    for (size_t s=0; s<scoreSnaps.size(); ++s) {
        auto &grid = scoreSnaps[s];
        int H = grid.size(), W = grid[0].size();
        // flatten
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                f << grid[i][j];
                if (!(i==H-1 && j==W-1)) f << ",";
            }
        }
        f << "\n";
    }
    f.close();
}

void write_totalScore_csv(const vector<vector<double>> &totalScore, const string &fname) {
    ofstream f(fname);
    int H = totalScore.size(), W = totalScore[0].size();
    for (int i=0;i<H;++i) {
        for (int j=0;j<W;++j) {
            f << totalScore[i][j];
            if (j < W-1) f << ",";
        }
        f << "\n";
    }
    f.close();
}


void write_ruleSnaps_csv(const vector<vector<vector<double>>> &ruleSnaps, const string &fname) {
    ofstream f(fname);
    // Each line: snap_index,y,x,ruleIndex,ruleValue   (sparse long form)
    int snaps = ruleSnaps.size();
    // For each snap
    for (int s=0; s<snaps; ++s) {
        //For each grid flatRules
        auto &flatRules = ruleSnaps[s]; // y -> x*ruleLen
        int y = (int)flatRules.size();
        int Xflat = (int)flatRules[0].size();
        // we don't know ruleLen directly; but it's Xflat / xLen. To keep things simple, output flattened full lines:
        // for each row write all values as a long comma-separated line (snap per line)
        // For row I
        for (int i=0;i<y;++i) {
            // for column J
            for (int j=0;j<Xflat;++j) {
                f << to_string(flatRules[i][j]);
                if (j<Xflat-1) f << ",";
            }
            f << "\n";
        }
    }
    f.close();
}

void write_nonCumulative_csv(const vector<vector<vector<double>>> &ncs, const string &fname) {
    ofstream f(fname);
    for (size_t s=0;s<ncs.size();++s) {
        auto &grid = ncs[s];
        int H = grid.size(), W = grid[0].size();
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                f << grid[i][j];
                if (!(i==H-1 && j==W-1)) f << ",";
            }
        }
        f << "\n";
    }
    f.close();
}

/* ---------------------------
main (testing)
--------------------------- */

int main(int argc, char** argv) {

    auto setUpStartTime = std::chrono::high_resolution_clock::now();
    if (argc < 6) {
        cerr << "Usage: ./sim p00 p01 p10 p11 gridN res0 res1 maxN rounds iters snaps evolutionRate mutationRate evolutionChance seed1 seed2 inversionpercent inversion round\n";
        return 1;
    }

    payoffMatrix = {
        {atof(argv[1]), atof(argv[2])}, //atof interprets the strings as floats
        {atof(argv[3]), atof(argv[4])}
    };


    int gridN = atoi(argv[5]); //atoi interterprets strings as integers
    pair<int,int> res = {atoi(argv[6]),atoi(argv[7])}; //what exactly is res?
    int maxN = atoi(argv[8]); //what is maxN? how does it relate to gridN
    int rounds = atoi(argv[9]);
    int iters = atoi(argv[10]);
    int snaps = atoi(argv[11]);

    double evolutionRate = atof(argv[12]); //what is evolution rate - it doesn't looke like it is ever used?
    double mutationRate = atof(argv[13]); //mutation rate randomly changes also the strategies a bit every round
    double evolutionChance = atof(argv[14]); //evolution Chance - change of adopting winner's strategy?
    unsigned int gridSeed = (unsigned) std::atoi(argv[15]); //randomness for distributing agents
    unsigned int playSeed = (unsigned) std::atoi(argv[16]); //randomness for playing
    // global_rng.seed(playSeed);
    // grid_rng.seed(gridSeed);
    double inversionPercentage = atof(argv[17]);//0; //what is inversion percentage and inversion round?
    int inversionRound = atoi(argv[18]);//1;

    int tileSize=atoi(argv[19]);
    std::cout<<"tile size is"<<tileSize;


    std::string path = argv[17];

    AgentGrid grid = blankGrid(gridN, res, gridSeed, mutationRate);


    TorusResult resu = torusTournament(grid, iters, rounds, snaps, evolutionRate, mutationRate, evolutionChance, inversionPercentage, inversionRound,tileSize);

    auto simulationEndTime = chrono::high_resolution_clock::now();

    write_scoreSnaps_csv(resu.scoreSnaps, path+"/scoreSnaps.csv");
    write_totalScore_csv(resu.totalScore, path+"/totalScore.csv");
    write_ruleSnaps_csv(resu.ruleSnaps, path+"/ruleSnaps.csv");
    write_nonCumulative_csv(resu.nonCumulativeScoreSnaps, path+"/nonCumulativeScore.csv");

    auto reportingEndTime = chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalElapsedTime = reportingEndTime - setUpStartTime;
    std::chrono::duration<double> setUpTime = setUpEndTime - setUpStartTime;
    std::chrono::duration<double> simulationTime = simulationEndTime - setUpEndTime;
    std::chrono::duration<double> reportingResultsTime = reportingEndTime - simulationEndTime;

    cout<< "TotalTime: " << totalElapsedTime.count()<<", SetUpTime: "<<setUpTime.count()<<", SimulationTime: "<<simulationTime.count()<<", TimePerRound: "<<simulationTime.count()/rounds<<", ReportingTime: "<<reportingResultsTime.count()<<endl;

    return 0;
}