th main.lua -expID final_model -LR 1.0e-5 -continue -addParallelSPPE -addSSTN -usePGPG -dataset ai-cha -GPU 2 -snapshot 1 -nEpochs 10 -trainIters 50000 -iterSize 2 -trainBatch 3 -validIters 4125 -validBatch 2 -nValidImgs 8250