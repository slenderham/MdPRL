function inputcheck = fCheckInput(expr, input)
%
% DESCRIPTION: checks probablities and sudo randomness of input
% combinations
%
% INPUT: 
% expr structure which includes number of trials for short and long blocks
% and input structure in check 
%
% OUTPUT:
% inputSide: resluts for checking probablities and sudo randomness of
% combinations
%
% Version History:
% 1.1:  [2015-09-13]

% double check input
sudorandom = mean(sort(reshape(input.inputCombination, expr.Ncombinations, [])),2) ;       % should be 1:18
inputcheck.inputCombination = mean(sudorandom==[1:expr.Ncombinations]')==1 ;
inputcheck.corr = corr(input.inputReward(1,:)', input.inputReward(2,:)') ;
inputcheck.prob_check0 = zeros(3,3,3) ;
inputcheck.prob_check1 = zeros(3,3,3) ;
choiceMapL = expr.choiceMap(1, :) ;
choiceMapR = expr.choiceMap(2, :) ;

for cnt_RL = 1:2
    inputSide = input.inputReward(cnt_RL,:) ;
    inputCombination = input.inputTarget(cnt_RL,:) ; 

    alignedinputSide_Long0 = inputSide(1:expr.NtrialsShort) ;
    alignedinputSide_Long1 = inputSide(1+expr.NtrialsShort:expr.Ntrials) ;

    alignedinputCombination_Long0 = inputCombination(1:expr.NtrialsShort) ;
%     alignedinputCombination_Long1 = inputCombination(1+expr.NtrialsShort:expr.Ntrials) ;

    for cnt_Ncombinations = expr.playcombinations
        idx_Ncombinations_Long0 = find(alignedinputCombination_Long0==cnt_Ncombinations) ;
        inputcheck.prob_check0(cnt_Ncombinations) = mean(alignedinputSide_Long0(idx_Ncombinations_Long0)) ;
        
%         idx_Ncombinations_Long1 = find(alignedinputCombination_Long1==cnt_Ncombinations) ;
%         inputcheck.prob_check1(cnt_Ncombinations) = mean(alignedinputSide_Long1(idx_Ncombinations_Long1)) ;
    end
    inputcheck.prob0(cnt_RL) = mean(mean(abs(inputcheck.prob_check0(expr.playcombinations)-expr.prob{1}(expr.playcombinations))<expr.precision))==1 ;
%     inputcheck.prob1(cnt_RL) = mean(mean(abs(inputcheck.prob_check1(expr.playcombinations)-expr.prob{2}(expr.playcombinations))<expr.precision))==1 ;
end

