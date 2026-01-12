clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../files")
addpath("../models")
addpath("../utils")

%%
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
% subjects1 = ["AA", "AB"];
subjects1 = lower(subjects1);
subjects1_inputs = "inputs/input_"+subjects1;
subjects1_prl = "SubjectData/PRL_"+subjects1;

subjects2 = [...
    "AA", "AB", "AC", "AD", "AE", "AG", ...
    "AH", "AI", "AJ", "AK", "AL", "AM", "AN", ...
    "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", ...
    "AW", "AX", "AY"] ;
% subjects2 = ["AA", "AB"] ;
subjects2_inputs = "inputs2/input_"+subjects2;
subjects2_prl = "SubjectData2/PRL_"+subjects2;

subjects_inputs = [subjects1_inputs subjects2_inputs];
subjects_prl = [subjects1_prl subjects2_prl];


exemplars = load('../files/RPL2Analysis_Exemplar.mat') ;
% attns = load('../files/RPL2Analysis_Attention_lim_temp_500_6models_40_rpe.mat') ;
ntrials = 432;

ntrialPerf       = 33:432;
% perfTH           = 0.5 + 2*sqrt(.5*.5/length(ntrialPerf)) ;
perfTH           = 0.53;

for cnt_sbj = 1:length(subjects_inputs)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

    load(inputname)
    load(resultsname)

    rew(cnt_sbj,:)                = results.reward ;
    [~, idxMax]                   = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew(cnt_sbj,:)          = results.choice' == idxMax ;
    perfMean(cnt_sbj)             = nanmean(choiceRew(cnt_sbj,ntrialPerf)) ;
    flaginfs(cnt_sbj) = expr.flaginf;
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
flaginfs = flaginfs(idxperf);
% idxperf = 1:length(subjects);

%%


parfor cnt_sbj = 1:length(subjects_inputs)
    disp(['Subject: ', num2str(cnt_sbj)])
    inputname   = strcat("../PRLexp/inputs_all/", subjects_inputs(cnt_sbj) , ".mat") ;
    resultsname = strcat("../PRLexp/SubjectData_all/", subjects_prl(cnt_sbj) , ".mat") ;
    
    inputs_struct = load(inputname);
    results_struct = load(resultsname);

    expr = results_struct.expr;
    input = inputs_struct.input;
    results = results_struct.results;
    
    expr.shapeMap = repmat([1 2 3 ;
                    1 2 3 ;
                    1 2 3 ], 1,1,3) ;

    expr.colorMap = repmat([1 1 1 ;
                    2 2 2 ;
                    3 3 3], 1,1,3) ;
                
    expr.patternMap(:,:,1) = ones(3,3) ;
    expr.patternMap(:,:,2) = 2*ones(3,3) ;
    expr.patternMap(:,:,3) = 3*ones(3,3) ;
    
    sesdata = struct();
    sesdata.input   = input ;
    sesdata.expr    = expr ;
    sesdata.results = results ;
    sesdata.NtrialsShort = expr.NtrialsShort ;
    sesdata.flagUnr = 1 ;
    sesdata.flag_couple = 0 ;
    sesdata.flag_updatesim = 0 ;

    NparamBasic = 4 ;
    if sesdata.flagUnr==1
        sesdata.Nalpha = 2 ;
    else
        sesdata.Nalpha = 1 ;
    end

    % load best params
    best_pars = exemplars.fit_results{2, cnt_sbj}.params;

    [trial_ll, latents] = fMLchoiceLL_exemplar(best_pars, sesdata);

    all_attns{cnt_sbj} = latents.A;
    trial_lls(cnt_sbj, :) = trial_ll;
    % trial_AICs(cnt_sbj, :) = 2*trial_lls(cnt_sbj, :)+2*length(best_pars)/ntrials;
    % trial_BICs(cnt_sbj, :) = 2*trial_lls(cnt_sbj, :)+log(ntrials)*length(best_pars)/ntrials;
    % trial_R2s(cnt_sbj, :) = 1-trial_BICs(cnt_sbj, :)./(2*log(2)+log(ntrials)*length(best_pars)/ntrials);
end

%%

all_attns_flat = reshape([all_attns{:}], [432, 3, 92]);
all_attns_flat = permute(all_attns_flat, [1,3,2]);

figure
clrmat = colormap('lines(3)');
clrmat = clrmat([2, 1, 3], :);
% posterior_model_ces = sum(all_model_attn_ws.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1]), [2 1 3 4 5]), [1 2]);
% posterior_model_ces = squeeze(posterior_model_ces);
attn_ws = all_attns_flat./sum(all_attns_flat,3);
wSize = 30;
smth_attn_ws = movmean(attn_ws, [0 wSize-1], 1, 'Endpoints', 'discard');
% smth_attn_ws = smoothdata(squeeze(all_model_attn_ws(5,3,:,:,:)), 2,"gaussian",wSize);
for d=[2 1 3]
%     plot_shaded_errorbar(squeeze(mean(smth_attn_ws(:,:,d), 1))', ...
%         squeeze(std(smth_attn_ws(:,:,d), [], 1))'/sqrt(length(idxperf)), ...
%         wSize, clrmat(d,:));hold on;

    plot_shaded_errorbar(squeeze(mean(smth_attn_ws(:,:,d),2)), ...
        std(smth_attn_ws(:,:,d),1,2)/sqrt(size(smth_attn_ws(:,:,d),2)), ...
        wSize, clrmat(d,:));hold on;
end

ylim([0.3, 0.38])
% yticks(0.:0.1:1.0)
xlim([wSize, ntrials])
xlabel('Trial')
ylabel('Normalized attn. weights')

legend(["", "Inf", "", "Noninf1", "", "Noninf2"],'Orientation','horizontal');

% 
% [clusters, p_values, t_sums, permutation_distribution ] = permutest(squeeze(smth_attn_ws(:,:,2))',...
% squeeze(smth_attn_ws(:,:,3))',false,0.05,10^3,true,inf);
% 
% disp(clusters)
% disp(p_values)
% 
% for num_cluster = 1:length(clusters)
%     if p_values(num_cluster)>0.05
%         continue
%     end
%     plot(clusters{num_cluster}+wSize-1, 0.21*ones(size(clusters{num_cluster})), ...
%         'MarkerSize', 10, 'MarkerEdgeColor',cmap(1,:), 'LineStyle', 'none', 'marker','.')
% end
% 
% [clusters, p_values, t_sums, permutation_distribution ] = permutest(squeeze(smth_attn_ws(:,:,1))',...
% squeeze(smth_attn_ws(:,:,3))',false,0.05,10^3,true,inf);
% 
% disp(clusters)
% disp(p_values)
% for num_cluster = 1:length(clusters)
%     if p_values(num_cluster)>0.05
%         continue
%     end
%     plot(clusters{num_cluster}+wSize-1, 0.22*ones(size(clusters{num_cluster})), ...
%         'MarkerSize', 10, 'MarkerEdgeColor',cmap(2,:), 'LineStyle', 'none', 'Marker','.')
% end
% 
% 
% legend(["", "Inf", "", "Noninf1", "", "Noninf2"],'Orientation','horizontal');
% 
