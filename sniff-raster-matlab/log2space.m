function ej=log2space(minValue,maxValue,numBins)

binEdgesLog2 = linspace(log2(minValue), log2(maxValue), numBins);
ej = 2.^binEdgesLog2;