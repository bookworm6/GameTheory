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

            for (int n = 0; n < iters; ++n) {
                unsigned long long a1prev = a1->prevMove;
                unsigned long long a2prev = a2->prevMove;
                int a1move = a1->playMove(a2prev, local_uniform01(), n);
                int a2move = a2->playMove(a1prev, local_uniform01(), n);
                // accumulate into thread-local arrays
                scoreTracker_local[idy-startY+1][idx-startX+1] += payoffMatrix[a1move][a2move];
                scoreTracker_local[localMatchCoordY-startY+1][localMatchCoordX-startX+1] += payoffMatrix[a2move][a1move];
            }
            a1->reset();
            a2->reset();
        }
    }
};