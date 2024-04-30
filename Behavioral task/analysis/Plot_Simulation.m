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
%% load result files
% feat = load('../files/RPL2Analysisv3_5_FeatureBased') ;
% obj = load('../files/RPL2Analysisv3_5_FeatureObjectBased') ;
% conj  = load('../files/RPL2Analysisv3_5_ConjunctionBased') ;
set(0,'defaultAxesFontSize',25)
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

all_model_names = ["fMLchoiceLL_RL2ftdecayattn", ...
    "fMLchoiceLL_RL2ftobjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn_onlyfattn", ...
    "fMLchoiceLL_RL2conjdecayattn_constrained"];

% make names for plotting
all_model_names_legend = ["F", "F+O", "F+C_{untied}", "F+C_{feat attn}", "F+C_{tied}"];
attn_modes_legend = strcat(attn_modes(:,1),"X",attn_modes(:,2));
[all_model_names_legend, attn_modes_legend] = meshgrid(all_model_names_legend, attn_modes_legend);
all_legends = strcat(all_model_names_legend(:), " ", attn_modes_legend(:));


ntrials = 432;
ntrialPerf       = 33:432;
% perfTH           = 0.5 + 2*sqrt(.5*.5/length(ntrialPerf)) ;
perfTH           = 0.53;
num_reps = 1;

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
    allperfMean(cnt_sbj)          = nanmean(choiceRew(cnt_sbj,:)) ;
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
% idxperf = 1:length(subjects);

%% load model fit
attns = load('../files/RPL2Analysis_Attention_merged_rep40_250.mat') ;

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            lls(m, a, cnt_sbj) = attns.fit_results{m, a, idxperf(cnt_sbj)}.fval;
            AICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+2*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
            BICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+log(ntrials)*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
        end
    end
end

[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
    mat2cell((1:50)', repmat([1], 1, 50)));
[~, best_model_inds] = max(g_BIC);

%% correlation between simulated performance and true performance

sim_m = 5;
sim_a = 3;

% clrmat = rgb('grey');
% fig = figure;
% t = tiledlayout(2,2);
% for i=1:length(sim_m)
%     for j=1:length(sim_a)
%         ax = nexttile;
scatter(perfMean(idxperf), squeeze(mean(all_sim_corrects(sim_m,sim_a,idxperf,:,:), [4,5])), 150, rgb('darkgreen')); hold on;
axis tight
ll=lsline;
errorbar(perfMean(idxperf), squeeze(mean(all_sim_corrects(sim_m,sim_a,idxperf,:,:), [4,5])), ...
    squeeze(std(mean(all_sim_corrects(sim_m,sim_a,idxperf,:,:), [5]),[], 4))/sqrt(num_reps), ...
    'LineStyle','none', 'Color', rgb('darkgreen'), 'LineWidth',1, 'CapSize', 0);
disp([corr(squeeze(mean(all_sim_corrects(sim_m,sim_a,idxperf,:,:), [4,5])), perfMean(idxperf)', 'type', 'pearson'), ...
      corr(squeeze(mean(all_sim_corrects(sim_m,sim_a,idxperf,:,:), [4,5])), perfMean(idxperf)', 'type', 'spearman')])
xticks(0.5:0.05:0.7);
yticks(0.5:0.05:0.7);
ylim([0.5, 0.68]);
xlim([0.51, 0.73]);
ll.LineWidth=1;
ll.Color=ones(1,3)*0.;
text(0.7, 0.55, '***', 'FontSize', 30)
% refline
% plot([0.48, 0.74], [0.48, 0.74], '--', 'Color', [0.6, 0.6, 0.6]);
% title(all_legends((sim_m-1)*10+sim_a));
%     end
% end

xlabel('True performance', 'FontSize', 30);
ylabel('Simulated performance', 'FontSize', 30);


% t.TileSpacing = 'compact';
% t.Padding = 'compact';

%% learning curves

% fig = figure;
% t = tiledlayout(2,2);
% for i=1:length(sim_m)
%     for j=1:length(sim_a)
%         ax = nexttile;
hline = plot_shaded_errorbar(squeeze(mean(movmean(all_sim_corrects(sim_m,sim_a,idxperf,1:nreps,:), 20, 5), [3 4])), ...
                     squeeze(std(movmean(all_sim_corrects(sim_m,sim_a,idxperf,1:nreps,:), 20, 5), [], [3 4]))/sqrt(length(idxperf)), ...
                     1, rgb('darkgreen'));hold on
hline.LineWidth = 2;
plot_shaded_errorbar(mean(movmean(choiceRew(idxperf,:), 20, 2))', ...
                     std(movmean(choiceRew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), ...
                     1, [0.5 0.5 0.5]);
xlim([0 ntrials+10])        
ylim([0.49, 0.67]);
% title(all_legends((sim_m-1)*10+sim_a));
%     end
% end

legend({'', 'Simulated', '', 'True'}, 'Location', 'southeast')
xlabel('Trial', 'Fontsize', 30)
ylabel('Percent correct', 'Fontsize', 30)
% t.TileSpacing = 'compact';
% t.Padding = 'compact';


% fig = figure;
% t = tiledlayout(2,2);
% for i=1:length(sim_m)
%     for j=1:length(sim_a)
%         ax = nexttile;
%         btsrp_corrs = bootstrp(500,@corr,squeeze(mean(all_sim_corrects(sim_m(i),sim_a(j),idxperf,:,:), [4,5])),perfMean(idxperf)');
%         histogram(btsrp_corrs, 'BinEdges', 0:0.05:1, 'FaceColor', clrmat((i-1)*2+j,:));
%         title(all_legends((sim_m(i)-1)*10+sim_a(j)));
%         xlim([0, 1])
%         ylim([0, 150])
%     end
% end

%% variability across subjective values

num_value_elems = 36;

fig=figure;

sim_values = reshape(cat(3, all_sim_values{sim_m, sim_a, idxperf, :}), [num_value_elems, ntrials, length(idxperf), num_reps]);
sim_values_std = std(sim_values, 1, 4); %normalized across runs for each subject
sim_values_cv = sim_values_std./(abs(mean(sim_values, 4)-0.5)+1e-2);
nexttile;
for k=1:9
    plot_shaded_errorbar(...
        squeeze(mean(sim_values_std(k,:,:),3))', ...
        squeeze(std(sim_values_std(k,:,:),1, 3))'/sqrt(length(idxperf)), ...
        1:ntrials, [0.8, 0.8, 0.8]); hold on;
end

for k=10:num_value_elems
    paxis = plot_shaded_errorbar(...
        squeeze(mean(sim_values_std(k,:,:),3))', ...
        squeeze(std(sim_values_std(k,:,:),1, 3))'/sqrt(length(idxperf)), ...
        1:ntrials, [0.8, 0.8, 0.8]); hold on;
    paxis.LineStyle = '--';
end
paxis_ft = plot_shaded_errorbar(...
        squeeze(mean(sim_values_std(1:9,:,:),[1 3]))', ...
        squeeze(std(sim_values_std(1:9,:,:),1,[1 3]))'/sqrt(length(idxperf)), ...
        1:ntrials, rgb('darkgreen')); hold on;
paxis_ft.DisplayName = "Feature";

paxis_conj = plot_shaded_errorbar(...
    squeeze(mean(sim_values_std(10:end,:,:),[1 3]))', ...
    squeeze(std(sim_values_std(10:end,:,:),1,[1 3]))'/sqrt(length(idxperf)), ...
    1:ntrials, rgb('darkgreen')); hold on;
paxis_conj.LineStyle = '--';
paxis_conj.DisplayName = "Conjunction";

ylim([0, 0.125])
xlim([0, ntrials])
legend([paxis_ft, paxis_conj], 'location', 'southeast')
%         title(all_legends((sim_m(i)-1)*10+sim_a(j)));
xlabel('Trial', 'FontSize', 28);
ylabel('Value variability', 'FontSize', 28);
% t.TileSpacing = 'compact';
% t.Padding = 'compact';


%% variability across subjective attentions

fig=figure;


sim_attns = reshape(cat(3, all_sim_attns{sim_m, sim_a, idxperf, :}), [2, 3, ntrials, length(idxperf), num_reps]);
sim_attns = squeeze(sim_attns(2,:,:,:,:));
mean_sim_attns = mean(sim_attns, 4);
sim_attns_jsd = mean(kl_div(sim_attns, mean_sim_attns, 1), 4); %normalized across runs for each subject
plot_shaded_errorbar(...
    squeeze(mean(sim_attns_jsd,3))', ...
    squeeze(std(sim_attns_jsd,1, 3))'/sqrt(length(idxperf)), ...
    1:ntrials, rgb('darkgreen')); hold on;
ylim([0, 1])
xlim([0, ntrials])

% legend(["", all_legends((sim_m(1)-1)*10+sim_a(2)), "", all_legends((sim_m(2)-1)*10+sim_a(2))], 'location', 'southeast', 'FontSize', 20)
xlabel('Trial', 'FontSize', 30);
ylabel('Attention variability', 'FontSize', 30);

%% correlation between attention and performance for simulation

fig=figure;
clrmat = colormap('lines(3)');
clrmat = clrmat([2, 1, 3], :);
% t = tiledlayout(1,2, 'TileSpacing','tight');
% for i=2
%     for j=2
sim_attns = reshape(cat(3, all_sim_attns{sim_m, sim_a, idxperf, :}), [2, 3, ntrials, length(idxperf), num_reps]);
sim_attns = squeeze(mean(sim_attns(2,:,:,:,:), 3));
trial_perfs = squeeze(all_sim_corrects(sim_m,sim_a,idxperf,:,:));
sim_idxperf = mean(trial_perfs(:,:,ntrialPerf),3)>perfTH;

sim_idxperf = find(sim_idxperf(:));
perfs = reshape(mean(trial_perfs, 3), [], 1);
perfs = perfs(sim_idxperf);
sim_attns = reshape(sim_attns, 3, []);
sim_attns = sim_attns(:, sim_idxperf);

for d = [2 1 3]
    plt(d) = scatter(reshape(squeeze(sim_attns(d,:)), [], 1), reshape(perfs, [], 1), 'filled', ...
            'Color', clrmat(d,:), 'MarkerFaceAlpha', 0.3); hold on;
    [r, p] = corr(reshape(perfs, [], 1), reshape(squeeze(sim_attns(d,:)), [], 1),'type','pearson');
    disp([r, p])
end
lsls = lsline();
lsls(1).Color = clrmat(3,:);
lsls(2).Color = clrmat(1,:);
lsls(3).Color = clrmat(2,:);
for d = 1:3
    lsls(d).LineWidth = 2;
end
ylim([0.51, 0.76])
xlim([-0.01, 1.12])
%     end
% end
legend(plt([2 1 3]), ["Inf", "Noninf1", "Noninf2"], 'location', 'best', 'Orientation','horizontal')

text(1.02, 0.64, "***", "fontsize", 30, "Color", clrmat(2,:))
text(1.02, 0.55, "***", "fontsize", 30, "Color", clrmat(1,:))
text(1.02, 0.565, "***", "fontsize", 30, "Color", clrmat(3,:))

% legend(["", all_legends((sim_m(1)-1)*10+sim_a(2)), "", all_legends((sim_m(2)-1)*10+sim_a(2))], 'location', 'southeast', 'FontSize', 20)
xlabel('Average attention weights', 'FontSize', 30);
ylabel('Performance', 'FontSize', 30);


%%

for cnt_sbj = 1:length(idxperf)
    avg_weights(cnt_sbj,:) = (attns.fit_results{m, a, idxperf(cnt_sbj)}.params(2)).*( ...
                                    attns.fit_results{m, a, idxperf(cnt_sbj)}.params(5)+ ...
                                    attns.fit_results{m, a, idxperf(cnt_sbj)}.params(6));
%     avg_weights(cnt_sbj,:) = 1;
    for cnt_rep = 1:nreps
        attn_where = 2;
         all_sim_model_ents(cnt_sbj, cnt_rep, :) = squeeze(entropy(all_sim_attns{sim_m, sim_a, idxperf(cnt_sbj)}(attn_where,:,:)));
         all_sim_model_jsds(cnt_sbj, cnt_rep, :) = squeeze(js_div( ...
            movmean(all_sim_attns{sim_m, sim_a, idxperf(cnt_sbj)}(attn_where,:,2:end), 1, 3), ...
            movmean(all_sim_attns{sim_m, sim_a, idxperf(cnt_sbj)}(attn_where,:,1:end-1), 1, 3), 2));
        for d = 1:3
            all_sim_model_attn_ws(cnt_sbj, cnt_rep, :, d) = squeeze(all_sim_attns{sim_m, sim_a, idxperf(cnt_sbj), cnt_rep}(attn_where,d,:));
        end
    end
    norm_avg_weights = avg_weights./sum(avg_weights, 1);
end

%%
figure;
% posterior_model_ents = sum(all_model_ents.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
% posterior_model_jsds = sum(all_model_jsds.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
colororder([rgb('purple'); rgb('navy')])
yyaxis left
wSize = 20;
smth_ents = movmean(squeeze(all_sim_model_ents), [0 wSize-1], 3, 'Endpoints', 'discard');
plot_shaded_errorbar(squeeze(mean(smth_ents, [1 2])), squeeze(std(smth_ents, [], [1 2]))/sqrt(length(idxperf)), 1:ntrials-wSize+1, rgb('purple'));hold on;
ylabel('Entropy')
yyaxis right
smth_jsds = movmean(squeeze(all_sim_model_jsds), [0 wSize-1], 3, 'Endpoints', 'discard');
plot_shaded_errorbar(squeeze(mean(smth_jsds, [1 2])), squeeze(std(smth_jsds, [], [1 2]))/sqrt(length(idxperf)), 1:ntrials-wSize, rgb('navy'));hold on;
ylabel('JSD')
% ylim([0, 0.3])
xlim([0, ntrials-wSize])
xlabel('Trial')

%%
trial_X = (1:ntrials)./ntrials;
trial_X = reshape(trial_X, [1 1 ntrials]);

all_data = [reshape(all_sim_model_ents, [], 1), ...
            reshape(repmat(trial_X, length(idxperf), num_reps, 1), [], 1), ...
            reshape(repmat((1:length(idxperf)), num_reps, ntrials), [], 1)];
all_data = array2table(all_data, "VariableNames", ["ents", "trial", "subj"]);
mdl = fitlme(all_data, 'ents~trial+(trial|subj)', 'CheckHessian',true);

[~,~,stats] = fixedEffects(mdl, 'DFMethod','Satterthwaite')
%%
trial_X = (1:ntrials-1)./ntrials;
trial_X = reshape(trial_X, [1 1 ntrials-1]);
all_data = [reshape(all_sim_model_jsds, [], 1), ...
            reshape(repmat(trial_X, length(idxperf), num_reps, 1), [], 1), ...
            reshape(repmat((1:length(idxperf)), num_reps, ntrials-1), [], 1)];
all_data = array2table(all_data, "VariableNames", ["jsds", "trial", "subj"]);
mdl = fitlme(all_data, 'jsds~trial+(trial|subj)',  'CheckHessian',true);
[~,~,stats] = fixedEffects(mdl, 'DFMethod','Satterthwaite')

%%
figure
clrmat = colormap('lines(3)');
clrmat = clrmat([2, 1, 3], :);
% posterior_model_ces = sum(all_model_attn_ws.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1]), [2 1 3 4 5]), [1 2]);
% posterior_model_ces = squeeze(posterior_model_ces);
wSize = 30;
smth_attn_ws = movmean(all_sim_model_attn_ws, [0 wSize-1], 3, 'Endpoints', 'discard');
smth_attn_ws = reshape(smth_attn_ws, num_reps*length(idxperf), ntrials-wSize+1, 3);
expanded_norm_avg_weights = reshape(repmat(norm_avg_weights, 1, num_reps), ...
    num_reps*length(idxperf), 1, 1);
% smth_attn_ws = squeeze(mean(smth_attn_ws, 2));
% smth_attn_ws = smoothdata(squeeze(all_model_attn_ws(5,3,:,:,:)), 2, 'movmean', [0 wSize-1]);
for d=[2 1 3]
%     plot_shaded_errorbar(squeeze(mean(smth_attn_ws(:,:,d), 1))', ...
%         squeeze(std(smth_attn_ws(:,:,d), [], 1))'/sqrt(length(idxperf)), ...
%         wSize, clrmat(d,:));hold on;
    plot_shaded_errorbar(squeeze(sum(smth_attn_ws(:,:,d).*expanded_norm_avg_weights/num_reps, 1))', ...
        std(bootstrp(1000, @(x, w) [sum(x.*w)./sum(w)], ...
            (smth_attn_ws(:,:,d)), expanded_norm_avg_weights(:,1)),[],1)', ...
        wSize, clrmat(d,:));hold on;
end

[clusters, p_values, t_sums, permutation_distribution ] = permutest(squeeze(smth_attn_ws(:,:,2))',...
squeeze(smth_attn_ws(:,:,3))',false,0.05,10^3,true,inf);

for num_cluster = 1:length(clusters)
    if p_values(num_cluster)>0.05
        continue
    end
    plot(clusters{num_cluster}+wSize-1, 0.21*ones(size(clusters{num_cluster})), ...
        'MarkerSize', 10, 'MarkerEdgeColor',cmap(1,:), 'LineStyle', 'none', 'marker','.')
end

[clusters, p_values, t_sums, permutation_distribution ] = permutest(squeeze(smth_attn_ws(:,:,1))',...
squeeze(smth_attn_ws(:,:,3))',false,0.05,10^3,true,inf);

for num_cluster = 1:length(clusters)
    if p_values(num_cluster)>0.05
        continue
    end
    plot(clusters{num_cluster}+wSize-1, 0.22*ones(size(clusters{num_cluster})), ...
        'MarkerSize', 10, 'MarkerEdgeColor',cmap(2,:), 'LineStyle', 'none', 'Marker','.')
end

legend(["", "Inf", "", "Noninf1", "", "Noninf2"], "Orientation", "Horizontal");
ylim([0.2, 0.55])
yticks(0.:0.1:1.0)
xlim([wSize, ntrials])
xlabel('Trial')
% ylabel('Attention weights')
ylabel('Effective attention weights')


%%
figure
plot3([0 0 1 0],[0 1 0 0],[1 0 0 1],'k', 'LineWidth', 1); hold on;
plot3([1/3 0], [1/3 0.5], [1/3 0.5], '--k');
plot3([1/3 0.5], [1/3 0], [1/3 0.5], '--k');
plot3([1/3 0.5], [1/3 0.5], [1/3 0], '--k');
xticks([])
yticks([])
zticks([])
axis equal
colormap viridis
view(120, 30)
% axis off
%plot A
scatter3(reshape(squeeze(mean(all_sim_model_attn_ws(:,:,:,1),3)), [], 1), ...
    reshape(squeeze(mean(all_sim_model_attn_ws(:,:,:,3),3)), [], 1), ...
    reshape(squeeze(mean(all_sim_model_attn_ws(:,:,:,2),3)), [], 1), ...
    40, reshape(squeeze(mean(all_sim_model_jsds,3)), [], 1), 'filled', 'MarkerFaceAlpha', 0.25)
cb = colorbar;
cb.Title.String = 'JSD';
cb.FontSize = 25;
text(1.2, 0.0, -0.1, 'Noninf1', 'FontSize',30)
text(-0.1, 0.1, 1., 'Inf', 'FontSize',30)
text(0.2, 0.9, -0.1, 'Noninf2', 'FontSize',30)
