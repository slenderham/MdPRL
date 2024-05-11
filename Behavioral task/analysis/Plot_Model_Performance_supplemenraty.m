clc
clear
close all
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../files")
addpath("../models")
addpath("../utils")
% addpath("../utils/DERIVESTsuite/DERIVESTsuite/")
% addpath("../utils/vbmc")

set(0,'defaultAxesFontSize',20)
%% load result files
% feat = load('../files/RPL2Analysisv3_5_FeatureBased') ;
% obj = load('../files/RPL2Analysisv3_5_FeatureObjectBased') ;
% conj  = load('../files/RPL2Analysisv3_5_ConjunctionBased') ;

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

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];

all_model_names = ["fMLchoiceLL_RL2conjdecayattn_constrained", ...
                   "fMLchoiceLL_RL2onlyconjdecayattn", ...
                   "fMLchoiceLL_RL2conjdecayattn_onlycattn"];
% all_model_names = all_model_names([1 2 3 4 6]);

% make names for plotting
all_model_names_legend = ["F+C_{joint}", "C_{attn}", "F+C_{conj attn}"];
attn_modes_legend = strcat(attn_modes(:,1),"X",attn_modes(:,2));
[all_model_names_legend, attn_modes_legend] = meshgrid(all_model_names_legend, attn_modes_legend);
all_legends = strcat(all_model_names_legend(:), "X", attn_modes_legend(:));


ntrials = 432;
ntrialPerf       = 33:432;
% perfTH           = 0.5 + 2*sqrt(.5*.5/length(ntrialPerf)) ;
perfTH           = 0.53;

cmap = lines(256);

for cnt_sbj = 1:length(subjects_inputs)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

    load(inputname)
    load(resultsname)

    rew(cnt_sbj,:)                = results.reward ;
    [~, idxMax]                   = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew(cnt_sbj,:)          = results.choice' == idxMax ;
    perfMean(cnt_sbj)             = nanmean(choiceRew(cnt_sbj,ntrialPerf)) ;
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
% idxperf = ~idxperf;
idxperf = find(idxperf);
% idxperf = 1:length(subjects);

%%
figure
plot_shaded_errorbar(mean(movmean(rew(idxperf,:), 20, 2))', std(movmean(rew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, 'k');
plot_shaded_errorbar(mean(movmean(choiceRew(idxperf,:), 20, 2))', std(movmean(choiceRew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, [0.5 0.5 0.5]);
ylim([0.4, 0.65])
quiver([86, 173, 259, 346, 432], ones(1,5)*0.4, zeros(1,5), ones(1,5)*0.01, "off", ...
    'Color','black', 'LineWidth', 2)
scatter([86, 173, 259, 346, 432], ones(1,5)*0.41, 40, 'k', 'filled', ...
    'Marker', '^')
xlabel('Trial')
ylabel('Performance')
legend({'', 'Reward', '', 'Proportion better'})
xlim([0, 432])


%% load results with attn and ML params

% attns = load('../files/RPL2Analysis_Attention_lim_temp_500_6models_40_pl_bounds.mat') ;
% attns = load('../files/RPL2Analysis_Attention_lim_temp_500_6models_40_merged.mat') ;
% attns = load('../files/RPL2Analysis_Attention_lim_temp_500_6models_40_rpe.mat') ;
attns = load('../files/RPL2Analysis_Attention_merged_rep40_500_log.mat');
attns_suppl = load('../files/RPL2Analysis_Attention_suppl.mat');

for a = 1:length(attn_modes)
    for cnt_sbj = 1:length(idxperf)
        num_params = length(attns.fit_results{5, a, idxperf(cnt_sbj)}.params);
        lls(1, a, cnt_sbj) = attns.fit_results{5, a, idxperf(cnt_sbj)}.fval;
        AICs(1, a, cnt_sbj) = 2*lls(1, a, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        BICs(1, a, cnt_sbj) = 2*lls(1, a, cnt_sbj)+log(ntrials)*num_params;
        Rsqs(1, a, cnt_sbj) = 1-BICs(1, a, cnt_sbj)./(-2*logsigmoid(0)*ntrials+log(ntrials)*num_params);
    end
end

for m = 2:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            num_params = length(attns_suppl.fit_results{m-1, a, idxperf(cnt_sbj)}.params);
            lls(m, a, cnt_sbj) = attns_suppl.fit_results{m-1, a, idxperf(cnt_sbj)}.fval;
            AICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
            BICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+log(ntrials)*num_params;
            Rsqs(m, a, cnt_sbj) = 1-BICs(m, a, cnt_sbj)./(-2*logsigmoid(0)*ntrials+log(ntrials)*num_params);
        end
    end
end

% disp(mean(Rsqs, 3))


% learning_strat = categorical(1:5)';
% attn_where = categorical([1 2 3 4 2 3 4 2 3 4]);
% attn_func = categorical([1 2 2 2 3 3 3 4 4 4]);
% 
% flat_BICs = reshape(BICs, [], 1);
% learning_strat = reshape(repmat(learning_strat, 1, 10, length(idxperf)), [], 1);
% attn_where = reshape(repmat(attn_where, 5, 1, length(idxperf)), [], 1);
% attn_func = reshape(repmat(attn_func, 5, 1, length(idxperf)), [], 1);
% 
% [ps, tbl] = anovan(flat_BICs, {learning_strat, attn_where}, "Varnames",["learning_strat","attn_where"], 'model',2);


%% plot differences in BICs
cmap = colormap('turbo(11)');
cmap(1,:) = 0.5;
bb = bar(mean(BICs-BICs(1,1,:), 3), 'FaceColor','flat'); hold on
xlim([0.5 5.5]);
yticks(-100:4:100);
for k = 1:10
    bb(k).CData = cmap(k,:);
end
xticklabels(all_model_names_legend(1,:)');
ylabel('\Delta BIC');
bbx = nan(length(attn_modes), length(all_model_names));
for i = 1:10
    bbx(i,:) = bb(i).XEndPoints;
end
errorbar(bbx',mean(BICs-BICs(1,1,:), 3),std(BICs-BICs(1,1,:), [], 3)./sqrt(numel(idxperf)),'k','linestyle','none');
legend(attn_modes_legend(:,1), 'location', 'eastoutside');

%% bayesian model selection

% [alpha_AIC,exp_r_AIC,xp_AIC,pxp_AIC,bor_AIC,g_AIC] = bms(reshape(-permute(AICs/2, [2 1 3]), [50, length(idxperf)])', ...
%     mat2cell((1:50)', repmat([1], 1, 50)));
% disp(bor_AIC);

[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(reshape(-permute(BICs/2, [2 1 3]), ...
    [length(all_model_names)*length(attn_modes), length(idxperf)])', ...
    mat2cell((1:length(all_model_names)*length(attn_modes))', repmat([1], 1, length(all_model_names)*length(attn_modes))));
disp(bor_BIC);
[~, best_model_inds] = max(g_BIC);


[alpha_input,exp_r_input,xp_input,pxp_input,bor_input,g_input] = bms(reshape(-permute(BICs/2, [2 1 3]), ...
    [length(all_model_names)*length(attn_modes), length(idxperf)])', ...
    mat2cell(reshape(1:length(all_model_names)*length(attn_modes), [length(attn_modes), length(all_model_names)])', repmat([1], 1, length(all_model_names))));
disp(bor_input);


[alpha_attn,exp_r_attn,xp_attn,pxp_attn,bor_attn,g_attn] = bms(reshape(-BICs/2, ...
    [length(all_model_names)*length(attn_modes), length(idxperf)])', ...
    mat2cell(reshape(1:length(all_model_names)*length(attn_modes), [length(all_model_names), length(attn_modes)])', repmat([1], 1, length(attn_modes))));
disp(bor_attn);

%% plot results
t = tiledlayout(50, 100, 'TileSpacing','tight');
nexttile([15 86])
imagesc(alpha_attn'/sum(alpha_attn));
txts = text((1:10)-0.35, ones(1, 10), string(num2str(pxp_attn(:), '%.2f')), 'FontSize',14);
for i=1:10
    if (alpha_attn(i)/sum(alpha_attn)>0.3)
        txts(i).Color = [1 1 1];
    end
end
% imagesc(pxp_attn)
caxis([0 1])
xticks([])
yticks([])
nexttile([15 14])
axis off
nexttile([35 86])
imagesc(reshape(alpha_BIC/sum(alpha_BIC), 10, 3)');
[txs, tys] = meshgrid(1:10, 1:length(all_model_names));
txs = txs';
tys = tys';
txts = text(txs(:)-0.35, tys(:), string(num2str(pxp_BIC(:), '%.2f')),'FontSize',14);
for i=1:length(all_model_names)
    for j=1:10
        if (alpha_BIC((i-1)*10+j)/sum(alpha_BIC)>0.3)
            txts((i-1)*10+j).Color = [1 1 1];
        end
    end
end
% imagesc(reshape(pxp_BIC, 10, 5)');
caxis([0 1])
xticks(1:10)
xticklabels(attn_modes_legend)
h=gca;
h.XAxis.TickLength = [0 0];
h.YAxis.TickLength = [0 0];
xlabel('Attentional mechanisms', 'Fontsize', 20)
yticks(1:length(all_model_names))
yticklabels({'F+C_{joint}', "C_{attn}", "F+C_{conj attn}"}')
xtickangle(30)
ylabel('Learning strategies', 'Fontsize', 20)
nexttile([35 14])
imagesc(alpha_input/sum(alpha_input));
% imagesc(pxp_input')
txts = text(ones(1, 3)-0.35, 1:length(all_model_names), string(num2str(pxp_input(:), '%.2f')), 'FontSize',14);
for i=1:length(all_model_names)
    if (alpha_input(i)/sum(alpha_input)>0.3)
        txts(i).Color = [1 1 1];
    end
end
caxis([0 1])
xticks([])
yticks([])
colormap(flipud(bone))
cb = colorbar;
cb.Layout.Tile = 'South';
cb.Label.String = 'Posterior model probability';
cb.Label.FontSize = 16;
t.TileSpacing = 'tight';
t.Padding = 'tight';
