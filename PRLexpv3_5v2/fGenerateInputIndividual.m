function [] = fGenerateInputIndividual(subName, flaginf)

rng('shuffle')
% setup           = fPRL2initSetup ;
expr            = fPRL2initExp(flaginf) ;
expr.flaginf    = flaginf ;
input           = fGenerateInput(expr) ;
save(['../PRLexp/inputs_sim/input_',subName,'.mat'],'expr','input');