clc
close all
clear

set(0,'defaultAxesFontSize',15)
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
subjects_prl = [subjects1_prl subjects2_prl];

ntrialPerf       = 33:432;
perfTH = 0.53;
idxSubject       = 1:length(subjects_inputs);
wSize  = 20 ;

for cnt_sbj = 1:length(subjects_inputs)
    disp(cnt_sbj)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj}, '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj}, '.mat'] ;

    load(inputname)
    load(resultsname)

    rew{cnt_sbj}        = results.reward ;
    [~, idxMax]         = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew{cnt_sbj}  = results.choice' == idxMax ;
    perfMean(cnt_sbj)   = nanmean(choiceRew{cnt_sbj}(ntrialPerf)) ;
end

idxperf = perfMean>=perfTH;
idxperf(29) = false;
idxperf(36) = false;
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
clear all_prob_ests
clear b1 b21 b22 b23 b3
for cnt_sbj = 1:length(subjects_inputs)
    inputname   = strcat('../PRLexp/inputs_all/', subjects_inputs(cnt_sbj) , '.mat') ;
    resultsname = strcat('../PRLexp/SubjectData_all/', subjects_prl(cnt_sbj) , '.mat') ;

    load(inputname)
    load(resultsname)

    expr.shapeMap     = repmat([1 2 3 ;
        1 2 3 ;
        1 2 3 ], 1,1,3) ;

    expr.colorMap     = repmat([1 1 1 ;
        2 2 2 ;
        3 3 3], 1,1,3) ;

    expr.patternMap(:,:,1)      = ones(3,3) ;
    expr.patternMap(:,:,2)      = 2*ones(3,3) ;
    expr.patternMap(:,:,3)      = 3*ones(3,3) ;

    for cnt_probe = 1:length(results.probEst)
        probEst         = results.probEst{cnt_probe} ;
        probEst         = probEst./(1+probEst) ;
%         probEst = log(probEst);

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

        % features
        Regft1 = mean(expr.prob{1}, [2 3]);
        Regft1 = repmat(Regft1, [1 3 3]);
        Regft2 = mean(expr.prob{1}, [1 3]);
        Regft2 = repmat(Regft2, [3 1 3]);
        Regft3 = mean(expr.prob{1}, [1 2]);
        Regft3 = repmat(Regft3, [3 3 1]);
        % Regft3 is constant so omitted

        % conjunctions
        Regconj1 = mean(expr.prob{1}, 1);
        Regconj1 = repmat(Regconj1, [3 1 1]);
%         Regconj1 = Regft1.*Regconj1 ./ (Regft1.*Regconj1+(1-Regft1).*(1-Regconj1));
        Regconj2 = mean(expr.prob{1}, 2);
        Regconj2 = repmat(Regconj2, [1 3 1]);
%         Regconj2 = Regft2.*Regconj2 ./ (Regft2.*Regconj2+(1-Regft2).*(1-Regconj2));
        Regconj3 = mean(expr.prob{1}, 3);
        Regconj3 = repmat(Regconj3, [1 1 3]);
%         Regconj3 = Regft3.*Regconj3 ./ (Regft3.*Regconj3+(1-Regft3).*(1-Regconj3));

        % objects
        LLRegobj = LLSh ;
        Regobj = LLRegobj./(1+LLRegobj) ;

        XTEMP = [Regft1(expr.playcombinations); Regft2(expr.playcombinations); ...
                 Regconj1(expr.playcombinations); Regconj2(expr.playcombinations); ...
                 Regconj3(expr.playcombinations); Regobj(expr.playcombinations); ...
                 probEst(expr.playcombinations)]' ;
        XTEMP = round(XTEMP./0.05)*0.05 ;
        XTEMP = log(XTEMP+1e-6) - log(1-XTEMP+1e-6);

%         XALL(cnt_probe, cnt_sbj, :, :) = XTEMP;

        X = XTEMP(:,end);
        Y1 = XTEMP(:, 1);
%         Y1 = [Y1 ones(size(Y1,1),1)] ;
        Y21 = XTEMP(:,[1 3]);
%         Y21 = [Y21 ones(size(Y21,1),1)] ;
        Y22 = XTEMP(:, 4);
%         Y22 = [Y22 ones(size(Y22,1),1)] ;
        Y23 = XTEMP(:, 5);
%         Y23 = [Y23 ones(size(Y23,1),1)] ;
        Y3 = XTEMP(:, 6);
%         Y3 = [Y3 ones(size(Y3,1),1)] ;

        mdl = fitlm(Y1, X);
        b1(:,cnt_probe,cnt_sbj) = mdl.Coefficients.Estimate;
        RsqS(1, cnt_probe,cnt_sbj) = mdl.Rsquared.Adjusted;
        lme(1, cnt_probe,cnt_sbj) = -mdl.LogLikelihood;
        
        mdl = fitlm(Y21, X);
        b21(:,cnt_probe,cnt_sbj) = mdl.Coefficients.Estimate;
        RsqS(2, cnt_probe,cnt_sbj) = mdl.Rsquared.Adjusted;
        lme(2, cnt_probe,cnt_sbj) = -mdl.LogLikelihood;
        
        mdl = fitlm(Y22, X);
        b22(:,cnt_probe,cnt_sbj) = mdl.Coefficients.Estimate;
        RsqS(3, cnt_probe,cnt_sbj) = mdl.Rsquared.Adjusted;
        lme(3, cnt_probe,cnt_sbj) = -mdl.LogLikelihood;
        
        mdl = fitlm(Y23, X);
        b23(:,cnt_probe,cnt_sbj) = mdl.Coefficients.Estimate;
        RsqS(4, cnt_probe,cnt_sbj) = mdl.Rsquared.Adjusted;
        lme(4, cnt_probe,cnt_sbj) = -mdl.LogLikelihood;

        mdl = fitlm(Y3, X);
        b3(:,cnt_probe,cnt_sbj) = mdl.Coefficients.Estimate;
        RsqS(5, cnt_probe,cnt_sbj) = mdl.Rsquared.Adjusted;
        lme(5, cnt_probe,cnt_sbj) = -mdl.LogLikelihood;

        all_prob_ests(:, cnt_probe, cnt_sbj) = probEst(expr.playcombinations);
%                 if any(isnan(probEst))
%           SS(cnt_sbj, cnt_probe, :) = nan*[tbl{2:end,2}];
%                 else
%         [p,tbl,stats,terms] = anovan(probEst(expr.playcombinations), ...
%                                     {expr.shapeMap(expr.playcombinations), ...
%                                      expr.colorMap(expr.playcombinations), ...
%                                      expr.patternMap(expr.playcombinations)}, ...
%                                 "model","interaction", ...
%                                 "varnames",["shape", "color", "pattern"],'display','off');
%             SS(cnt_sbj, cnt_probe, :) = [tbl{2:end,2}];
%                 end
    end
end

%% population level anova
for cnt_probe = 1:length(results.probEst)
    [p,tbl,stats,terms] = anovan(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), 1, []), ...
                                {repmat(expr.shapeMap(expr.playcombinations), [1, sum(idxperf)]), ...
                                 repmat(expr.colorMap(expr.playcombinations), [1, sum(idxperf)]), ...
                                 repmat(expr.patternMap(expr.playcombinations), [1, sum(idxperf)])}, ...
                                "model","full", ...
                                "varnames",["shape", "color", "pattern"],'display','off');
    anova_ps(cnt_probe, :) = p;
    eta2(cnt_probe, :) = ([tbl{2:end-2,2}]-[tbl{2:end-2,3}].*[tbl{end-1,5}])./([tbl{end,2}]+[tbl{end-1,5}]);
end

%% 

figure
clrmats = colormap('turbo(6)');
clrmats = clrmats(1:5, :);
for i=1:5
errorbar(nanmean(RsqS(i,:,:), 3)', nanstd(RsqS(i,:,:), [], 3)'/sqrt(size(RsqS, 3)), 'Color', clrmats(i,:), 'LineWidth', 1);hold on;
end
ylim([0, 0.425])
legend(["F_{inf}", "F_{inf}+C_{inf}", "C_{noninf1}", "C_{noninf2}", "O"], "Location", "northwest")
xticks(1:5)
xlabel('Value Estimation Trial')
ylabel('R^2')
xlim([0.5, 5.5])

% figure
% prob_est_rmses = (XALL(:,:,:,4)-reshape(expr.prob{1}(expr.playcombinations), [1 1 27])).^2;
% plot_shared_errorbar(nanmean(prob_est_rmses, [2 3]), nanstd(prob_est_rmses, [], [2 3])/sqrt(size(XALL, 2)));

figure;
plot(eta2(:,[2 1 3 5 6 4 7]), '-o', 'LineWidth', 1); hold on;
xlim([0.5, 5.5])
ylim([-0.05, 0.3])
xticks(1:5)
xlabel('Value Estimation Trial')
ylabel('\omega^2')
legend(["F_{inf}", "F_{noninf1}", "F_{noninf2}", "C_{inf}", "C_{noninf1}", "C_{noninf2}", "O"], "Location", "northwest");

% SS = SS(:,:,1:7)./SS(:,:,9);