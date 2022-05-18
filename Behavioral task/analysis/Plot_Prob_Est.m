clc 
close all
clear
%%

addpath("../files")
addpath("../models")
addpath("../utils")

subjects1 = [...
    "AA", "AB", "AC", "AD", "AE", "AF", "AG", ...
    "AH", "AI", "AJ", "AK", "AL", "AM", "AN", ...
    "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", ...
    "AW", "AX", "AY", "AZ", "BA", "BB", "BC", "BD", ...
    "BE", "BF", "BG", "BH", "BI", "BJ", "BK", "BL", ...
    "BM", "BN", "BO", "BP", "BQ", "BR", "CC", "DD", ...
    "EE", "FF", "GG", "HH", "II", "JJ", "KK", "LL", ...
    "MM", "NN", "OO", "PP", "QQ", "RR", "SS", "TT", ...
    "UU", "VV", "WW", "XX", "YY", "ZZ"];
subjects1 = lower(subjects1);
subjects1_inputs = "inputs/input_"+subjects1;
subjects1_prl = "SubjectData/PRL_"+subjects1;

subjects2 = [...
    "AA", "AB", "AC", "AD", "AE", "AG", ...
    "AH", "AI", "AJ", "AK", "AL", "AM", "AN", ...
    "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", ...
    "AW", "AX", "AY"] ;
subjects2_inputs = "inputs2/input_"+subjects2;
subjects2_prl = "SubjectData2/PRL_"+subjects2;

subjects_inputs = [subjects1_inputs subjects2_inputs];
subjects_prl = [subjects1_prl subjects2_prl];

ntrialPerf       = 33:432;
perfTH           = 0.5 + 2*sqrt(.5*.5/length(ntrialPerf)) ;
perfTH = 0.53;
idxSubject       = 1:length(subjects_inputs);
wSize            = 20 ; 

for cnt_sbj = 1:length(subjects_inputs)
    disp(cnt_sbj)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj}, '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj}, '.mat'] ;
    
    load(inputname)
    load(resultsname)
    
    rew{cnt_sbj}                  = results.reward ;
    [~, idxMax]                   = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew{cnt_sbj}            = results.choice' == idxMax ;
    perfMean(cnt_sbj)             = nanmean(choiceRew{cnt_sbj}(ntrialPerf)) ;
end

idxperf = perfMean>perfTH;
idxperf(29) = false;
subjects_inputs = subjects_inputs(idxperf);
subjects_prl = subjects_prl(idxperf);

%% 

probeTrialsAll = load(['../PRLexp/inputs_all/inputs/input_', 'aa' , '.mat']).expr.trialProbe;

for cnt_probe = 1:length(probeTrialsAll)
    pEstAll{cnt_probe}   = nan*ones(length(subjects_inputs),27) ;
    XAll{cnt_probe}      = [] ;
end

clear sesdata
clear SS
for cnt_sbj = 1:length(subjects_inputs)
    inputname   = strcat('../PRLexp/inputs_all/', subjects_inputs(cnt_sbj) , '.mat') ;
    resultsname = strcat('../PRLexp/SubjectData_all/', subjects_prl(cnt_sbj) , '.mat') ;

    load(inputname)
    load(resultsname)

    expr.shapeMap               = repmat([1 2 3 ;
                                          1 2 3 ;
                                          1 2 3 ], 1,1,3) ;

    expr.colorMap               = repmat([1 1 1 ;
                                          2 2 2 ;
                                          3 3 3], 1,1,3) ;
                                
    expr.patternMap(:,:,1)      = ones(3,3) ;
    expr.patternMap(:,:,2)      = 2*ones(3,3) ;
    expr.patternMap(:,:,3)      = 3*ones(3,3) ;

    for cnt_probe = 1:length(results.probEst)
        probEst         = results.probEst{cnt_probe} ;
        probEst         = probEst./(1+probEst) ;

        % regression coefficient
        X   = 4 ; 
        ll1 = [X      1  1/X;
               1/X^2  1  X^2;
               X      1  1/X]' ;

        X   = 3 ; 
        ll  = [X      1  1/X]' ;
        
        LLSh            = nan*ones(3,3,3) ;
        LLSh(1,:,:)     = ll1*ll(1) ;
        LLSh(2,:,:)     = ones(size(ll1))*ll(2) ;
        LLSh(3,:,:)     = ll1*ll(3) ;
        
        % informative feature
        llRegft = [1  1  1;
                   1  1  1;
                   1  1  1]' ;
        LLRegft          = nan*ones(3,3,3) ;
        LLRegft(1,:,:)   = llRegft*ll(1) ;
        LLRegft(2,:,:)   = llRegft*ll(2) ;
        LLRegft(3,:,:)   = llRegft*ll(3) ;
        Regft            = LLRegft./(1+LLRegft) ;
        
        % conjunction
        X = 4;
        llRegcnj = [X      1  1/X;
                    1/X^2  1  X^2;
                    X      1  1/X]' ;
        LLRegcnj         = nan*ones(3,3,3) ;
        LLRegcnj(1,:,:)  = llRegcnj.^(2/3)*1 ;
        LLRegcnj(2,:,:)  = llRegcnj.^(2/3)*1 ;
        LLRegcnj(3,:,:)  = llRegcnj.^(2/3)*1 ;
        Regcnj           = LLRegcnj./(1+LLRegcnj) ;
        
         % objects
        LLRegobj         = LLSh ;
        Regobj           = LLRegobj./(1+LLRegobj) ;
        

        XTEMP                              = [Regft(expr.playcombinations); Regcnj(expr.playcombinations); Regobj(expr.playcombinations); probEst(expr.playcombinations)]' ; 
        XTEMP                              = round(XTEMP./0.05)*0.05 ;
        XAll{cnt_probe}                    = [XAll{cnt_probe}; XTEMP] ;

%         if any(isnan(probEst))
%             SS(cnt_sbj, cnt_probe, :) = nan*[tbl{2:end,2}];
%         else
        [p,tbl,stats,terms] = anovan(probEst(expr.playcombinations), ...
            {expr.shapeMap(expr.playcombinations), expr.colorMap(expr.playcombinations), expr.patternMap(expr.playcombinations)}, ...
            "model","interaction","varnames",["shape", "color", "pattern"],'display','off');
        SS(cnt_sbj, cnt_probe, :) = [tbl{2:end,2}];
%         end
    end
end

figure;
errorbar(squeeze(nanmean(SS(:,:,1:6)./SS(:,:,8),1)), squeeze(nanstd(SS(:,:,1:6)./SS(:,:,8),[], 1))/sqrt(length(idxperf)), '-o')
xlim([0.5, 5.5])
ylim([0.05, 0.5])
xticks(1:5)
xlabel('Value Estimation Trial')
ylabel('Percent of Variance Explained')
legend(["shape", "color", "pattern", "shapeXcolor", "shapeXpattern", "colorXpattern"], "Location", "northwest");

% SS = SS(:,:,1:7)./SS(:,:,9);