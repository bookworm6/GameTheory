#include <iostream>
#include "RandomGeneratorPCG/pcg_random.hpp"

#ifndef RNGHALFBITS
#define RNGHALFBITS
class rngHalfBits {

    private: 
        uint32_t normalize; //this will be the value of 16 bits of ones. it will be computed at compile time. 
        int firstHalf;
        int secondHalf;
        uint32_t currentRandomNumber;
        bool generateNew;
        pcg32 local_rng;

    public:
        rngHalfBits():
            local_rng(20){
            normalize = ((uint32_t)(-1))>>16; //this will be the value of 16 bits of ones. it will be computed at compile time. 
            printBits(normalize);
            std::cout<<"normalize"<<normalize;
            firstHalf = 0;
            secondHalf = 0;
            currentRandomNumber = 0;
            generateNew=true;
            
        }
    
    

        float generate (){
            if (generateNew){
                currentRandomNumber = local_rng();
                firstHalf = currentRandomNumber>>16;
                secondHalf = (currentRandomNumber<<16)>>16;
                generateNew = false;
                return ((float)firstHalf)/((float)normalize); 
            }
            else{
                generateNew = true;
                return ((float)secondHalf)/((float)normalize); 
            }
        }
    private:
    void printBits (int toPrint){
        std::cout << std::bitset<sizeof(toPrint) * 8>(toPrint) << "   ";
    }
    
};

#endif 
