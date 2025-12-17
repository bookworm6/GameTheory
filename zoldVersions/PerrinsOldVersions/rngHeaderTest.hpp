#include <iostream>
#include "RandomGenerator/Xoshiro.hpp"

#ifndef RNGHALFBITS
#define RNGHALFBITS


class rngHalfBits {

    private: 
        uint64_t normalize; //this will be the value of 16 bits of ones. it will be computed at compile time. 
        uint64_t currentRandomNumber;
        bool generateNew;
        int bitsAvailable;
        XoshiroCpp::Xoroshiro128Plus local_rng;

    public:
        rngHalfBits():
            local_rng(20){
            normalize = ((uint64_t)(-1))>>48; //this will be the value of 16 bits of ones. it will be computed at compile time. 
            currentRandomNumber = 0;
            bitsAvailable=0;
            
        }

        float generate (){
            if (bitsAvailable==0){
                currentRandomNumber = local_rng();
                bitsAvailable = 64;
            }
            int toReturn = ((uint64_t)currentRandomNumber)>>(bitsAvailable-16);
            bitsAvailable-=16;
            int shiftAmount = 64-bitsAvailable;
            currentRandomNumber = ((uint64_t)(currentRandomNumber<<shiftAmount))>>shiftAmount;
            return (float)toReturn/(float)normalize;
        }
    private:
    void printBits (uint64_t toPrint){
        std::cout << std::bitset<sizeof(toPrint) * 8>(toPrint) << "   ";
    }
    
};

#endif 
