function input = fGenerateInput(expr)
%
% DESCRIPTION: Generates input for expreriment; 4 schedules and 6 total
% combinations 
%
% INPUT: 
% expr structure which includes number of short/long blocks, number
% of trials, and probabilities
%
% OUTPUT:
% inputSide: side which reward is assigned
% inputCombination: input combinations
%
% Version History:
% 1.1:  [2015-09-13]

% generate input
inputRewardR        = [] ;
inputRewardL        = [] ;
inputTargetR        = [] ;
inputTargetL        = [] ;
inputCombination    = [] ;
N_blocksAll         = [] ;
choiceMapL          = expr.choiceMap(1, :) ;
choiceMapR          = expr.choiceMap(2, :) ;

% here we have only 1 block (block is when probabilities change)
for cnt_Nblocks = 1:expr.Nblocks
    
    % to make sure subjects see each combination once before seeing them
    % for another time
    N_blocksAll             = [N_blocksAll cnt_Nblocks*ones(1,expr.NtrialsShort)] ;
    inputCombination_blocks = [] ;
    for cnt_sudo = 1:(expr.NtrialsShort/expr.Ncombinations)
        inputCombination_blocks = [inputCombination_blocks randperm(expr.Ncombinations)] ;
    end
    
    % combination to target
    inputTargetR_blocks = choiceMapR(inputCombination_blocks) ;
    inputTargetL_blocks = choiceMapL(inputCombination_blocks) ;
    
    % assign probabilities sudo randomly
    for cnt_sudo2 = 1:(expr.NtrialsShort/expr.NtrialsRewardAssign)
        idx_trial = (cnt_sudo2-1)*expr.NtrialsRewardAssign+1:cnt_sudo2*expr.NtrialsRewardAssign ;
        for cnt_Ncombinations       = expr.playcombinations
            idxR_Ncombinations      = find(inputTargetR_blocks(idx_trial)==cnt_Ncombinations) ;
            NR_Ncombinations        = length(idxR_Ncombinations) ;
            Nprob_NcombinationsR    = round(expr.prob{cnt_Nblocks}(cnt_Ncombinations)*length(idxR_Ncombinations)) ;
            SideR                   = [ones(1, Nprob_NcombinationsR) zeros(1, NR_Ncombinations-Nprob_NcombinationsR)] ;

            idxL_Ncombinations      = find(inputTargetL_blocks(idx_trial)==cnt_Ncombinations) ;
            NL_Ncombinations        = length(idxL_Ncombinations) ;
            Nprob_NcombinationsL    = round(expr.prob{cnt_Nblocks}(cnt_Ncombinations)*length(idxL_Ncombinations)) ;
            SideL                   = [ones(1, Nprob_NcombinationsL) zeros(1, NL_Ncombinations-Nprob_NcombinationsL)] ;

            % to make sure reward assignment is not /anti correlated
            for cnt_corr = 1:50
                inputSideR_blocksShort(idxR_Ncombinations, cnt_corr)    = SideR(randperm(NR_Ncombinations)) ;
                inputSideL_blocksShort(idxL_Ncombinations, cnt_corr)    = SideL(randperm(NL_Ncombinations)) ;
            end
        end
        corr_blocksShort        = abs(corr(inputSideR_blocksShort, inputSideL_blocksShort)) ;
        [idx_corrR, idx_corrL]  = find(corr_blocksShort==min(corr_blocksShort(:))) ;
        idx_rand                = randi(length(idx_corrR)) ;
        inputRewardR            = [inputRewardR; inputSideR_blocksShort(:,idx_corrR(idx_rand))] ;
        inputRewardL            = [inputRewardL; inputSideL_blocksShort(:,idx_corrL(idx_rand))] ;
    end
    inputCombination            = [inputCombination inputCombination_blocks] ;
    inputTargetR                = [inputTargetR inputTargetR_blocks] ;
    inputTargetL                = [inputTargetL inputTargetL_blocks] ;
end
input.inputReward           = [inputRewardL inputRewardR]'  ;
input.inputCombination      = inputCombination ;
input.inputTarget           = [inputTargetL; inputTargetR] ;
input.Nschedule_blocksShortAll = N_blocksAll ;



