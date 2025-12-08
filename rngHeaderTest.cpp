#include "rngHeaderTest.hpp"
#include <iostream>
#include <fstream>
std::ofstream outputFile("generatedRandom.txt");
int main(){
    rngHalfBits generator;
    for (int i=0; i<1000; i++){
        outputFile<<generator.generate()<<", ";
    }
    outputFile.close();
    return 0;
}