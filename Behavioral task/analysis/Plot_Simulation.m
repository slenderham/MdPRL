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

set(0,'defaultAxesFontSize',22)
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

sim_m = [1, 5];
sim_a = [1, 3];

clrmat = [rgb('deepskyblue');rgb('blue');rgb('limegreen');rgb('green')];

fig = figure;
t = tiledlayout(2,2);
for i=1:length(sim_m)
    for j=1:length(sim_a)
        ax = nexttile;
        scatter(squeeze(mean(all_sim_corrects(sim_m(i),sim_a(j),idxperf,:,:), [4,5])), perfMean(idxperf), 50, clrmat((i-1)*2+j,:)); hold on;
        disp([corr(squeeze(mean(all_sim_corrects(sim_m(i),sim_a(j),idxperf,:,:), [4,5])), perfMean(idxperf)', 'type', 'pearson'), ...
              corr(squeeze(mean(all_sim_corrects(sim_m(i),sim_a(j),idxperf,:,:), [4,5])), perfMean(idxperf)', 'type', 'spearman')])
        xlim([0.48, 0.74]);
        ylim([0.48, 0.74]);
        xticks(0.5:0.1:0.7);
        yticks(0.5:0.1:0.7);
        ll=lsline;
        ll.LineWidth=1;
        ll.Color=ones(1,3)*0.5;
        plot([0.48, 0.74], [0.48, 0.74], '--k');
        title(all_legends((sim_m(i)-1)*10+sim_a(j)));
    end
end

xlabel(t,'Simulated Performance', 'FontSize', 20);
ylabel(t,'True Performance', 'FontSize', 20);
t.TileSpacing = 'compact';
t.Padding = 'compact';

%% learning curves

fig = figure;
t = tiledlayout(2,2);
for i=1:length(sim_m)
    for j=1:length(sim_a)
        ax = nexttile;
        hline = plot_shaded_errorbar(squeeze(mean(movmean(all_sim_corrects(sim_m(i),sim_a(j),idxperf,1:nreps,:), 20, 5), [3 4])), ...
                             squeeze(std(movmean(all_sim_corrects(sim_m(i),sim_a(j),idxperf,1:nreps,:), 20, 5), [], [3 4]))/sqrt(length(idxperf)), ...
                             1, clrmat((i-1)*2+j,:));hold on
        hline.LineWidth = 2;
        plot_shaded_errorbar(mean(movmean(choiceRew(idxperf,:), 20, 2))', ...
                             std(movmean(choiceRew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), ...
                             1, [0.5 0.5 0.5]);
        xlim([0 ntrials+10])        
        ylim([0.49, 0.69]);
        title(all_legends((sim_m(i)-1)*10+sim_a(j)));
    end
end

xlabel(t, 'Trial', 'Fontsize', 20)
ylabel(t, 'Percent Correct', 'Fontsize', 20)
t.TileSpacing = 'compact';
t.Padding = 'compact';


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

num_value_elems = [9, 36];

fig=figure;
t = tiledlayout(2,2);
for i=1:length(sim_m)
    for j=1:length(sim_a)
        sim_values = reshape(cat(3, all_sim_values{sim_m(i), sim_a(j), idxperf, :}), [num_value_elems(i), ntrials, length(idxperf), 50]);
        sim_values_std = std(sim_values, 1, 4); %normalized across runs for each subject
        sim_values_cv = sim_values_std./(abs(mean(sim_values, 4)-0.5)+1e-2);
        nexttile;
        for k=1:num_value_elems(i)
            plot_shaded_errorbar(...
                squeeze(mean(sim_values_std(k,:,:),3))', ...
                squeeze(std(sim_values_std(k,:,:),1, 3))'/sqrt(length(idxperf)), ...
                1:ntrials, [0.8, 0.8, 0.8]); hold on;
        end
        plot_shaded_errorbar(...
                squeeze(mean(sim_values_std(1:9,:,:),[1 3]))', ...
                squeeze(std(sim_values_std(1:9,:,:),1,[1 3]))'/sqrt(length(idxperf)), ...
                1:ntrials, clrmat((i-1)*2+j,:)); hold on;
        if i==2
            paxis = plot_shaded_errorbar(...
                squeeze(mean(sim_values_std(10:end,:,:),[1 3]))', ...
                squeeze(std(sim_values_std(10:end,:,:),1,[1 3]))'/sqrt(length(idxperf)), ...
                1:ntrials, clrmat((i-1)*2+j,:)); hold on;
            paxis.LineStyle = '--';
        end
        ylim([0, 0.135])
        title(all_legends((sim_m(i)-1)*10+sim_a(j)));
    end
end

xlabel(t,'Trial', 'FontSize', 20);
ylabel(t,'Value Variability', 'FontSize', 20);
t.TileSpacing = 'compact';
t.Padding = 'compact';


%% variability across subjective attentions

fig=figure;
clrmat = [rgb('deepskyblue');rgb('blue');rgb('limegreen');rgb('green')];
for i=1:length(sim_m)
    for j=2
        sim_attns = reshape(cat(3, all_sim_attns{sim_m(i), sim_a(j), idxperf, :}), [2, 3, ntrials, length(idxperf), 50]);
        sim_attns = squeeze(sim_attns(2,:,:,:,:));
        mean_sim_attns = mean(sim_attns, 4);
        sim_attns_jsd = mean(kl_div(sim_attns, mean_sim_attns, 1), 4); %normalized across runs for each subject
        plot_shaded_errorbar(...
            squeeze(mean(sim_attns_jsd,3))', ...
            squeeze(std(sim_attns_jsd,1, 3))'/sqrt(length(idxperf)), ...
            1:ntrials, clrmat((i-1)*2+j,:)); hold on;
        ylim([0, 1])
        xlim([0, ntrials])
    end
end

legend(["", all_legends((sim_m(1)-1)*10+sim_a(2)), "", all_legends((sim_m(2)-1)*10+sim_a(2))], 'location', 'southeast', 'FontSize', 20)
xlabel('Trial', 'FontSize', 20);
ylabel('Attention Variability', 'FontSize', 20);

%% correlation between attention and performance for simulation

fig=figure;
clrmat = colormap('lines(3)');
clrmat = clrmat([2, 1, 3], :);
% t = tiledlayout(1,2, 'TileSpacing','tight');
for i=2
    for j=2
        sim_attns = reshape(cat(3, all_sim_attns{sim_m(i), sim_a(j), idxperf, :}), [2, 3, ntrials, length(idxperf), 50]);
        sim_attns = squeeze(mean(sim_attns(2,:,:,:,:), 3));
        perfs = squeeze(mean(all_sim_corrects(sim_m(i),sim_a(j),idxperf,:,:), 5));
        for d = [2 1 3]
            plt(d) = scatter(reshape(squeeze(sim_attns(d,:,:)), [], 1), reshape(perfs, [], 1), 'filled', ...
                    'Color', clrmat(d,:), 'MarkerFaceAlpha', 0.3); hold on;
            [r, p] = corr(reshape(perfs, [], 1), reshape(squeeze(sim_attns(d,:,:)), [], 1),'type','pearson');
            disp([r, p])
        end
        lsls = lsline();
        lsls(1).Color = clrmat(3,:);
        lsls(2).Color = clrmat(1,:);
        lsls(3).Color = clrmat(2,:);
        for d = 1:3
            lsls(d).LineWidth = 2;
        end
        ylim([0.39, 0.75])
        xlim([-0.05, 1.1])
    end
end
legend(plt([2 1 3]), ["Inf", "Noninf1", "Noninf2"], 'location', 'best', 'Orientation','horizontal')

% legend(["", all_legends((sim_m(1)-1)*10+sim_a(2)), "", all_legends((sim_m(2)-1)*10+sim_a(2))], 'location', 'southeast', 'FontSize', 20)
xlabel('Average Attention', 'FontSize', 20);
ylabel('Performance', 'FontSize', 20);


%%
for m = 5
    for a = 3
        for cnt_sbj = 1:length(idxperf)
            mean_attns = zeros(ntrials, 3);
            mean_ents = 0;
            mean_kls = 0;
            for cnt_rep = 1:nreps
                if strcmp(attn_modes(a,2), 'C')
                    attn_where = 1;
                elseif strcmp(attn_modes(a,2), 'L')
                    attn_where = 2;
                elseif strcmp(attn_modes(a,2), 'CL')
                    attn_where = 2;
                else
                    attn_where = 1;
                end
                 mean_ents = mean_ents + 1/nreps*squeeze(entropy(all_sim_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,:)));
                 mean_kls = mean_kls + 1/nreps*squeeze(symm_kl_div(all_sim_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,2:end), ...
                    all_sim_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,1:end-1), 2));

                for d = 1:3
                    mean_attns(:,d) = mean_attns(:,d) + squeeze(all_sim_attns{m, a, idxperf(cnt_sbj), cnt_rep}(attn_where,d,:))/nreps;
                end
            end
            all_sim_model_ents(m, a, cnt_sbj, :) = mean_ents;
            all_sim_model_kls(m, a, cnt_sbj, :) = mean_kls;
            all_sim_model_ces(m, a, cnt_sbj, :, :) = mean_attns;
        end
    end
end

figure
clrmat = colormap('winter(3)');
wSize = 1;
smth_ces = movmean(squeeze(all_sim_model_ces(1,3,:,:,:)), [0 wSize-1], 2, 'Endpoints', 'discard');
for d=[2 1 3]
    plot_shaded_errorbar(squeeze(mean(smth_ces(:,:,d), 1))', squeeze(std(smth_ces(:,:,d), [], 1))'/sqrt(length(idxperf)), wSize, clrmat(d,:));hold on;
end
legend(["", "inf", "", "noninf1", "", "noninf2"]);
ylim([0.15, 0.6])
xlim([wSize, ntrials+10])
xlabel('Trial')
ylabel('Attention Weights')

figure
clrmat = colormap('lines(3)');
wSize = 1;
smth_ces = movmean(squeeze(all_sim_model_ces(5,3,:,:,:)), [0 wSize-1], 2, 'Endpoints', 'discard');
for d=[2 1 3]
    plot_shaded_errorbar(squeeze(mean(smth_ces(:,:,d), 1))', squeeze(std(smth_ces(:,:,d), [], 1))'/sqrt(length(idxperf)), wSize, clrmat(d,:));hold on;
end
legend(["", "inf", "", "noninf1", "", "noninf2"]);
ylim([0.15, 0.6])
xlim([wSize, ntrials+10])
xlabel('Trial')
ylabel('Attention Weights')

figure
plot_shaded_errorbar(squeeze(mean(all_sim_model_ents(5,3,:,:,:), 3)), squeeze(std(all_sim_model_ents(5,3,:,:,:), [], 3))/sqrt(length(idxperf)), 1, rgb('deepskyblue'));hold on;
plot_shaded_errorbar(squeeze(mean(all_sim_model_kls(5,3,:,:,:), 3)), squeeze(std(all_sim_model_kls(5,3,:,:,:), [], 3))/sqrt(length(idxperf)), 1:ntrials-1, rgb('coral'));hold on;
legend(["", "Entropy", "", "KL_{symm}"])
ylim([0, 15])
xlim([0, ntrials+10])
xlabel('Trial')
figure
plot_shaded_errorbar(squeeze(mean(all_sim_model_ents(1,3,:,:,:), 3)), squeeze(std(all_sim_model_ents(1,3,:,:,:), [], 3))/sqrt(length(idxperf)), 1, rgb('deepskyblue'));hold on;
plot_shaded_errorbar(squeeze(mean(all_sim_model_kls(1,3,:,:,:), 3)), squeeze(std(all_sim_model_kls(1,3,:,:,:), [], 3))/sqrt(length(idxperf)), 1:ntrials-1, rgb('coral'));hold on;
legend(["", "Entropy", "", "KL_{symm}"])
ylim([0, 15])
xlim([0, ntrials+10])
xlabel('Trial')
