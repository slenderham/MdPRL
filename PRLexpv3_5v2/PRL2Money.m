clc
clear
close all
% rng('shuffle')

%%

load('./inputs/input_ag.mat')
load('./SubjectData/PRL_ag.mat')

perf = mean(results.reward);

%%

inputcheck = fCheckInput(expr, input) ;
expr.prob_check0 = inputcheck.prob_check0 ;
expr.prob_check1 = inputcheck.prob_check1 ;
perf_chanceActualP = mean(mean(expr.prob{1}(input.inputTarget))) ;
perf_maxActualP = mean(max(expr.prob{1}(input.inputTarget))) ; 

%%

MaxMoney = 1.5*20 ;
MinMoney = 1.5*10 ;
MoneyExtra = (perf-perf_chanceActualP)./(perf_maxActualP-perf_chanceActualP)*(MaxMoney-MinMoney)
MinMoney