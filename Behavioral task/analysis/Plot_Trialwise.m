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

    % "fMLchoiceLL_RL2conjdecayattn_spread", ...

% make names for plotting
all_model_names_legend = ["F", "F+O", "F+C_{untied}", "F+C_{feat attn}", "F+C_{tied}"];
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
    flaginfs(cnt_sbj) = expr.flaginf;
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
flaginfs = flaginfs(idxperf);
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

%% Simulate model with best param
% load('../files/simulated_vars')

unfilt_AICs = trial_AICs;
unfilt_BICs = trial_BICs;
unfilt_lls = trial_lls;

trial_AICs=trial_AICs(:,:,idxperf,:);
trial_BICs=trial_BICs(:,:,idxperf,:);
trial_lls=trial_lls(:,:,idxperf,:);

%% Plot Model Evidence

wSize = 100;
clrmat = colormap('lines(5)');

smth_AIC = movmean(trial_AICs, [0 wSize-1], 4, 'Endpoints', 'discard');
smth_BIC = movmean(trial_BICs, [0 wSize-1], 4, 'Endpoints', 'discard');
% smth_BIC = smoothdata(trial_BICs, 4, "gaussian", wSize);
smth_ll = movmean(trial_lls, [0 wSize-1], 4, 'Endpoints', 'discard');

figure
ylabel('Trial-wise \Delta BIC', 'FontSize', 27)
l_input = [];
for i=1:4
%     [~, min_attn_type] = max(alpha_BIC((i-1)*length(attn_modes)+1:i*length(attn_modes)));
%     if i==2
    min_attn_type = 3;
%     end
    l_input(i) = plot_shaded_errorbar(squeeze(mean(smth_BIC(5,3,:,:)-smth_BIC(i,min_attn_type,:,:), [1 2 3])), ...
        squeeze(std(smth_BIC(5,3,:,:)-smth_BIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(length(idxperf)), ...
        wSize, clrmat(i,:));hold on
end
yticks(-0.06:0.02:0.06)
axis tight
ylim([-0.04, 0.04]);
% xlim([0, ntrials-wSize+1]);
yline(0., ":")
legend(l_input, ["F", "F+O", "F+C_{separate}", "F+C_{feat attn}"], "Location", "northeast")
xlabel('Trial')


%%
model_diff_bics = squeeze(trial_BICs(5,3,:,:)-trial_BICs(2,3,:,:));
all_data = [reshape(model_diff_bics', [], 1), ...
            repmat((1:ntrials)'./ntrials, length(idxperf), 1), ...
            repelem((1:length(idxperf))', ntrials)];
all_data = array2table(all_data, "VariableNames", ["diff_bics", "trial", "subj"]);
% mdl = fitlme(all_data, 'diff_bics~trial+(trial-1|subj)' ...
%     ,'CheckHessian', true); 
% does not converge
mdl = fitlm(all_data, 'diff_bics~trial')

% 
% figure
% % clrmat_pre = clrmat_pre(2:4,:);
% clrmat = [0 0 0; colormap('hsv(4)')];
% ylabel('Trial-wise \Delta BIC', 'FontSize', 18)
% models_to_comp = [1 2 4 6 9];
% l_attn = [];
% for i_model=1:length(models_to_comp)
% %     [~, min_attn_type] = max(alpha_BIC((i-1)*length(attn_modes)+1:i*length(attn_modes)));
% %     if i==2
%     i = models_to_comp(i_model);
%     min_input_type = 5;
% %     end
%     l_attn(i_model) = plot_shaded_errorbar(squeeze(mean(smth_BIC(5,3,:,:)-smth_BIC(min_input_type,i,:,:), [1 2 3])), ...
%         squeeze(std(smth_BIC(5,3,:,:)-smth_BIC(min_input_type,i,:,:), [], [1 2 3]))/sqrt(length(idxperf)), ...
%         1:ntrials-wSize+1, clrmat(i_model,:));hold on
% end
% yticks(-0.1:0.01:0.02)
% ylim([-0.06, 0.02]);
% xlim([-10, ntrials-wSize+11]);
% yline(0., ":")
% legend(l_attn, attn_modes_legend(models_to_comp), "Location", "eastoutside")
% xlabel('Trial')

% figure
% for i=1:5
% %     [~, min_attn_type] = max(alpha_AIC((i-1)*length(attn_modes)+1:i*length(attn_modes)));
%     disp(min_attn_type)
%     l(i) = plot_shaded_errorbar(squeeze(mean(smth_AIC(i,min_attn_type,:,:), [1 2 3])), ...
%                                 squeeze(std(smth_AIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(length(idxperf)), ...
%                                 wSize, clrmat(i,:));hold on
% end
% legend(l, ["F", "F+O", "F+C_{feat attn}", "F+C_{untied}", "F+C_{tied}"])
% xlabel('Trial')
% ylabel('Trial-wise AIC')


%% quantify effect of attention
%% sharpness of attention (entropy)?

m_to_plot = 5;
a_to_plot = 2;

pre_smth_wsize = 1;

clear all_model_ents
clear all_model_jsds
for m = m_to_plot
    for a = a_to_plot
        for cnt_sbj = 1:length(idxperf)
            if strcmp(attn_modes(a,2), 'C')
                attn_where = 1;
            elseif strcmp(attn_modes(a,2), 'L')
                attn_where = 2;
            elseif strcmp(attn_modes(a,2), 'CL')
                attn_where = 2;
            else
                attn_where = 1;
            end
            if m~=3
                all_model_ents(m, a, cnt_sbj, :) = squeeze(entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,:)));
                all_model_jsds(m, a, cnt_sbj, :) = squeeze(js_div( ...
                    movmean(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,2:end), pre_smth_wsize, 3, 'Endpoints','discard'), ...
                    movmean(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,1:end-1), pre_smth_wsize, 3, 'Endpoints','discard'), 2));
            else
                all_model_ents(m, a, cnt_sbj, :) = squeeze((entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,:)) ...
                    +entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,:)))/2);
                all_model_jsds(m, a, cnt_sbj, :) = squeeze((js_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,2:end), ...
                    all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,1:end-1), 2) ...
                    +js_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,2:end), ...
                    all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,1:end-1), 2))/2);
            end
        end
    end
end

% fig=  figure;
% tiledlayout(3,3);
% for a = 2:length(attn_modes)
%     nexttile;
%     for m = 1:length(all_model_names)
%         plot_shaded_errorbar(squeeze(mean(all_model_ents(m,a,:,:), 3)), squeeze(std(all_model_ents(m,a,:,:), [], 3))/sqrt(length(idxperf)), 1, clrmat(m,:));hold on;
%         ylim([0, 1.2])
%         if mod(a, 3)==2
%             ylabel(attn_modes(a, 1))
%         end
%         if (a-1)/3>2
%             xlabel(attn_modes(a, 2))
%         end
%     end
% end

% lg  = legend(reshape([repmat([""], 5, 1) all_model_names_legend(1,:)']', 10, 1));
% lg.Layout.Tile = 'East';
% fax=axes(fig,'visible','off');
% fax.XLabel.Visible='on';
% fax.YLabel.Visible='on';
% ylabel(fax,'Attn Type');
% xlabel(fax,'Attn Time');

figure;
% posterior_model_ents = sum(all_model_ents.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
% posterior_model_jsds = sum(all_model_jsds.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
colororder([rgb('purple'); rgb('navy')])
yyaxis left
wSize = 20;
smth_ents = movmean(squeeze(all_model_ents(m_to_plot,a_to_plot,:,:,:)), [0 wSize-1], 2, 'Endpoints', 'discard');
plot_shaded_errorbar(squeeze(mean(smth_ents, 1))', squeeze(std(smth_ents, [], 1))'/sqrt(length(idxperf)), 1:ntrials-wSize+1, rgb('purple'));hold on;
ylabel('Entropy')
yyaxis right
smth_jsds = movmean(squeeze(all_model_jsds(m_to_plot,a_to_plot,:,:,:)), [0 wSize-1], 2, 'Endpoints', 'discard');
plot_shaded_errorbar(squeeze(mean(smth_jsds, 1))', squeeze(std(smth_jsds, [], 1))'/sqrt(length(idxperf)), 1:ntrials-wSize-pre_smth_wsize+1, rgb('navy'));hold on;
ylabel('JSD')
% ylim([0, 0.3])
xlim([0, ntrials-wSize])
xlabel('Trial')

%%
model_ents = squeeze(all_model_ents(m_to_plot,a_to_plot,:,:));
all_data = [reshape(model_ents', [], 1), ...
            repmat((1:ntrials)'./ntrials, length(idxperf), 1), ...
            repelem((1:length(idxperf))', ntrials)];
all_data = array2table(all_data, "VariableNames", ["ents", "trial", "subj"]);
mdl = fitlme(all_data, 'ents~trial+(trial|subj)', 'CheckHessian',true)
[~,~,stats] = fixedEffects(mdl, 'DFMethod','Satterthwaite')

%%
model_jsds = squeeze(all_model_jsds(m_to_plot,a_to_plot,:,:));
all_data = [reshape(model_jsds', [], 1), ...
            repmat((1:ntrials-1)'./(ntrials-1), length(idxperf), 1), ...
            repelem((1:length(idxperf))', ntrials-1)];
all_data = array2table(all_data, "VariableNames", ["jsds", "trial", "subj"]);
mdl = fitlme(all_data, 'jsds~trial+(trial|subj)', 'CheckHessian',true)
[~,~,stats] = fixedEffects(mdl, 'DFMethod','Satterthwaite')

% figure;
% correlation between ents, jsd, and model fit (sharper attention -> better
% fit for attentional model)
% correlation between ents, jsd, and accuracy (shouldn't be)


%% focus on the feature (avg attn weight)

for m = m_to_plot
    for a = a_to_plot
        for cnt_sbj = 1:length(idxperf)
            if strcmp(attn_modes(a,2), 'C')
                attn_where = 1;
            elseif strcmp(attn_modes(a,2), 'L')
                attn_where = 2;
            elseif strcmp(attn_modes(a,2), 'CL')
                attn_where = 2;
            else
                attn_where = 1;
            end
            avg_weights(cnt_sbj,:) = ...
                (attns.fit_results{m, a, idxperf(cnt_sbj)}.params(2)).*...
                (attns.fit_results{m, a, idxperf(cnt_sbj)}.params(5)+attns.fit_results{m, a, idxperf(cnt_sbj)}.params(6));
%             avg_weights(cnt_sbj,:) = 1;
            for d = 1:3
                if m~=3
                    all_model_attn_ws(m, a, cnt_sbj, :, d) = squeeze(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,d,:));
                else
                    conj_d = [2, 1, 3];
                    conj_d = conj_d(d);
                    all_model_attn_ws(m, a, cnt_sbj, :, d) = squeeze((all_attns{m, a, idxperf(cnt_sbj)}(attn_where,d,:) ...
                        +all_attns{m, a, idxperf(cnt_sbj)}(attn_where,3+conj_d,:))/2);
                end
            end
        end
        norm_avg_weights = avg_weights./sum(avg_weights, 1);
    end
end

%% plot average by subject
figure
clrmat = colormap('lines(3)');
clrmat = clrmat([2, 1, 3], :);
% posterior_model_ces = sum(all_model_attn_ws.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1]), [2 1 3 4 5]), [1 2]);
% posterior_model_ces = squeeze(posterior_model_ces);
attn_ws = squeeze(all_model_attn_ws(m_to_plot,a_to_plot,:,:,:));
wSize = 30;
smth_attn_ws = movmean(attn_ws, [0 wSize-1], 2, 'Endpoints', 'discard');
% smth_attn_ws = smoothdata(squeeze(all_model_attn_ws(5,3,:,:,:)), 2,"gaussian",wSize);
for d=[2 1 3]
%     plot_shaded_errorbar(squeeze(mean(smth_attn_ws(:,:,d), 1))', ...
%         squeeze(std(smth_attn_ws(:,:,d), [], 1))'/sqrt(length(idxperf)), ...
%         wSize, clrmat(d,:));hold on;

    plot_shaded_errorbar(squeeze(sum(smth_attn_ws(:,:,d).*norm_avg_weights(:,1), 1))', ...
        std(bootstrp(1000, @(x, w) [sum(x.*w)./sum(w)], ...
            (smth_attn_ws(:,:,d)), norm_avg_weights(:,1)),[],1)', ...
        wSize, clrmat(d,:));hold on;
end

ylim([0.2, 0.5])
yticks(0.:0.1:1.0)
xlim([wSize, ntrials])
xlabel('Trial')
ylabel('Effective attention weights')

[clusters, p_values, t_sums, permutation_distribution ] = permutest(squeeze(smth_attn_ws(:,:,2))',...
squeeze(smth_attn_ws(:,:,3))',false,0.05,10^3,true,inf);

disp(clusters)
disp(p_values)

for num_cluster = 1:length(clusters)
    if p_values(num_cluster)>0.05
        continue
    end
    plot(clusters{num_cluster}+wSize-1, 0.21*ones(size(clusters{num_cluster})), ...
        'MarkerSize', 10, 'MarkerEdgeColor',cmap(1,:), 'LineStyle', 'none', 'marker','.')
end

[clusters, p_values, t_sums, permutation_distribution ] = permutest(squeeze(smth_attn_ws(:,:,1))',...
squeeze(smth_attn_ws(:,:,3))',false,0.05,10^3,true,inf);

disp(clusters)
disp(p_values)
for num_cluster = 1:length(clusters)
    if p_values(num_cluster)>0.05
        continue
    end
    plot(clusters{num_cluster}+wSize-1, 0.22*ones(size(clusters{num_cluster})), ...
        'MarkerSize', 10, 'MarkerEdgeColor',cmap(2,:), 'LineStyle', 'none', 'Marker','.')
end


legend(["", "Inf", "", "Noninf1", "", "Noninf2"],'Orientation','horizontal');

%% plot single subject

figure
clrmat = colormap('lines(3)');
clrmat = clrmat([2, 1, 3], :);

attn_ws = squeeze(all_model_attn_ws(m_to_plot,a_to_plot,:,:,:));

subj_to_plot = randi(67);
for d=[2 1 3]
    plot(squeeze(all_model_attn_ws(5, 3, subj_to_plot, :, d)), ...
        'Color', clrmat(d,:), 'linewidth', 1);
    hold on
end
xlim([0, 432])
ylim([-0.05, 1.05])

xlabel('Trial')
ylabel('Attention weights')

box off



%% plot average by time
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
% axis([0 1 0 1 0 1])
view(120,30)
%plot A
scatter3(squeeze(mean(all_model_attn_ws(m_to_plot,a_to_plot,:,:,1),4)), ...
    squeeze(mean(all_model_attn_ws(m_to_plot,a_to_plot,:,:,3),4)), ...
    squeeze(mean(all_model_attn_ws(m_to_plot,a_to_plot,:,:,2),4)), ...
    40, squeeze(mean(all_model_jsds(m_to_plot,a_to_plot,:,:),4)), 'filled')
cb = colorbar;
cb.Title.String = 'JSD';
text(1.1, 0.0, -0.1, 'Noninf1', 'FontSize',25)
text(-0.1, -0.1, 1., 'Inf', 'FontSize',25)
text(0.2, 1.01, -0.05, 'Noninf2', 'FontSize',25)

%%
figure;
wSize = 432;
for d=[2 1 3]
%     plt(d) = scatter(reshape(squeeze(movmean(all_model_attn_ws(5,3,:,:,d),wSize,4,'endpoint','discard')), [], 1), ...
%                   reshape(movmean(choiceRew(idxperf,:),wSize,2,'endpoint','discard'), [], 1), 'filled', ...
%                     'Color', clrmat(d,:), 'MarkerEdgeAlpha', 1, 'MarkerFaceAlpha', 1);hold on;
    plt(d) = scatter(squeeze(mean(all_model_attn_ws(m_to_plot,a_to_plot,:,:,d), 4)), perfMean(idxperf)', 'filled', ...
                    'Color', clrmat(d,:), 'MarkerEdgeAlpha', 1, 'MarkerFaceAlpha', 1);hold on;
    [r, p] = corr(reshape(squeeze(movmean(all_model_attn_ws(m_to_plot,a_to_plot,:,:,d),wSize,4,'endpoint','discard')), [], 1), ...
                  reshape(movmean(choiceRew(idxperf,:),wSize,2,'endpoint','discard'), [], 1), 'type', 'spearman');
    disp([r, p])
end
lsls = lsline();
lsls(1).Color = clrmat(3,:);
lsls(2).Color = clrmat(1,:);
lsls(3).Color = clrmat(2,:);

legend(plt([2 1 3]), ["Inf", "Noninf1", "Noninf2"], 'Orientation','horizontal')
for i=1:3
    lsls(i).LineWidth = 2;
end
ylim([0.51, 0.77])
xlim([-0.05, 1.1])

text(1.02, 0.68, "*", "fontsize", 30, "Color", clrmat(2,:))
text(1.02, 0.51, "*", "fontsize", 30, "Color", clrmat(1,:))
ylabel("Performance")
xlabel("Average attention weights")

% figure;
% for d=[2 1 3]
%     plt(d) = plot(g_BIC(43,:), squeeze(mean(all_model_attn_ws(5,3,:,:,d),4)), '.', ...
%         'MarkerSize', 20, 'Color', clrmat(d,:));hold on;
%     lsls = lsline();
% end
% 
% legend(plt([2 1 3]), ["Inf", "Noninf1", "Noninf2"])
% for i=1:3
%     lsls(i).LineWidth = 1;
% end
% xlim([0, 1])
% ylim([-0.1, 1.1])
% xlabel("Posterior Prob of F+C_{tied} DiffXL")
% ylabel("Average attentional weights")



%%



%% differential signals in value (pairwise difference in values in different dimensions)
%% significance of "complementary" attention

channel_groups = {[0 3 6 9], ...
    [0 3 6 9 36], ...
    [0 3 6 9 18 27 36], ...
    [0 3 6 9 18 27 36], ...
    [0 3 6 9 18 27 36]};

rew_probs = 0.05*round(expr.prob{1}/0.05);
marginals{1} = reshape(squeeze(mean(rew_probs, [1, 3])), 3, 1);
marginals{2} = reshape(squeeze(mean(rew_probs, [2, 3])), 3, 1);
marginals{3} = reshape(squeeze(mean(rew_probs, [1, 2])), 3, 1);
marginals{4} = squeeze(mean(rew_probs, 1));
marginals{4} = marginals{4}(:);
marginals{5} = squeeze(mean(rew_probs, 2));
marginals{5} = marginals{5}(:);
marginals{6} = squeeze(mean(rew_probs, 3));
marginals{6} = marginals{6}(:);
marginals{7} = expr.prob{1}(:);

marginals_by_cg = {[1 2 3], ...
    [1 2 3 7], ...
    [1 2 3 4 5 6], ...
    [1 2 3 4 5 6], ...
    [1 2 3 4 5 6]};

%%
for m = 5
    disp(m);
    for a = 3
        for cnt_sbj = 1:length(idxperf)
            for l = 1:ntrials
                cg = channel_groups{m};
                all_value_pdists(m, a, cnt_sbj, l, 1:6) = nan;
                for cg_idx = 1:length(cg)-1
                    curr_vals = all_values{m, a, idxperf(cnt_sbj)}(cg(cg_idx)+1:cg(cg_idx+1), l);
                    curr_marginals = marginals{marginals_by_cg{m}(cg_idx)};
                    all_value_pdists(m, a, cnt_sbj, l, cg_idx) = std(curr_vals);
%                     temp_corr = (curr_vals-curr_vals').*sign(curr_marginals-curr_marginals');
%                     all_values_differentials(m, a, cnt_sbj, l, cg_idx) = ...
%                         mean(temp_corr(triu(true(size(temp_corr)), 1)));
                end
            end
        end
    end
end

% posterior_model_pdists = nansum(all_value_pdists.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1 1]), [2 1 3 4 5]), [1 2]);
% posterior_model_pdists = squeeze(posterior_model_pdists);

%%
figure;
clrmat=colormap('lines(6)');
clrmat = clrmat([2 1 3 4 5 6], :);

t = tiledlayout(2,1,'Padding','Compact', 'TileSpacing','compact');

nexttile
for i=[2 1 3]
    plot_shaded_errorbar(squeeze(nanmean(all_value_pdists(5,3,:,:,i), [1 3])), squeeze(nanstd(all_value_pdists(5,3,:,:,i), [], [1 3]))/sqrt(length(idxperf)), 1:ntrials, clrmat(i,:));
    xlim([0, 432])
end
legend(["", "F_{inf}", "", "F_{noninf1}", "", "F_{noninf2}"], ...
    'Location','southeast','Orientation','horizontal');
xticklabels([])

nexttile
for i=[4 5 6]
    plot_shaded_errorbar(squeeze(nanmean(all_value_pdists(5,3,:,:,i), [1 3])), squeeze(nanstd(all_value_pdists(5,3,:,:,i), [], [1 3]))/sqrt(length(idxperf)), 1:ntrials, clrmat(i,:));
    xlim([0, 432])
end
ylim([0, 0.085])
legend(["", "C_{inf}", "", "C_{noninf1}", "", "C_{noninf2}"], ...
    'Location','southeast','Orientation','horizontal');

ylabel(t, 'Value separability', 'fontsize', 25)
xlabel(t, 'Trial', 'fontsize', 25)

%% effect of reward and no-reward on value differential

all_attn_logits = zeros(length(all_model_names), length(attn_modes), length(idxperf), 1+ntrials, length(cg));
for m = 5
    disp(m);
    cg = channel_groups{m};
    for a = 2:length(attn_modes)
        % find out in which array the attetion is stored
        if strcmp(attn_modes(a,2), 'C')
            attn_where = 1;
        elseif strcmp(attn_modes(a,2), 'L')
            attn_where = 2;
        elseif strcmp(attn_modes(a,2), 'CL')
            attn_where = 2;
        else
            attn_where = 1;
        end
        % loop over subjects
        for cnt_sbj = 1:length(idxperf)
            resultsname = strcat("../PRLexp/SubjectData_all/", subjects_prl(cnt_sbj) , ".mat") ;
            results_struct = load(resultsname);
            %             curr_subj_C = results_struct.results.choice;
            curr_subj_R = results_struct.results.reward; % 1-ntrials
            % loop over trials
            for l = 1:ntrials
                % for each trial, find the winning dimension
                for cg_idx = 1:length(cg)-1
                    curr_vals = all_values{m, a, idxperf(cnt_sbj)}(cg(cg_idx)+1:cg(cg_idx+1), l);
                    if a==1
                        all_attn_logits(m, a, cnt_sbj, l, cg_idx) = 0;
                    elseif 2<=a && a<=4
                        all_attn_logits(m, a, cnt_sbj, l, cg_idx) = mean(pdist(curr_vals));
                    elseif 5<=a && a<=7
                        all_attn_logits(m, a, cnt_sbj, l, cg_idx) = mean(curr_vals);
                    elseif 8<=a && a<=10
                        all_attn_logits(m, a, cnt_sbj, l, cg_idx) = mean(squareform(...
                            bsxfun(@(x,y) max(x,y), curr_vals, curr_vals').*(1-eye(length(curr_vals)))));
                    end
                end
                if m==1 || m==2 || m==4
                    [~, curr_attn_I] = sort(squeeze(all_attns{m, a}(attn_where,:,l)), 'descend');
                    all_attn_logits_diff(m, a, cnt_sbj, l-1, 1:3) = ...
                        all_attn_logits(m, a, cnt_sbj, l, curr_attn_I) ...
                        - all_attn_logits(m, a, cnt_sbj, l-1, curr_attn_I);
                elseif m==3
                    [~, curr_attn_I] = sort(squeeze(all_attns{m, a}(attn_where,1:3,l)), 'descend');
                    all_attn_logits_diff(m, a, cnt_sbj, l-1, 1:3) = ...
                        all_attn_logits(m, a, cnt_sbj, l, curr_attn_I) ...
                        - all_attn_logits(m, a, cnt_sbj, l-1, curr_attn_I);
                    [~, curr_attn_I] = sort(squeeze(all_attns{m, a}(attn_where,4:6,l)), 'descend');
                    all_attn_logits_diff(m, a, cnt_sbj, l-1, 4:6) = ...
                        all_attn_logits(m, a, cnt_sbj, l, curr_attn_I) ...
                        - all_attn_logits(m, a, cnt_sbj, l-1, curr_attn_I);
                elseif m==5
                    [~, curr_attn_I] = sort(squeeze(all_attns{m, a}(attn_where,[2 1 3],l)), 'descend');
                    all_attn_logits_diff(m, a, cnt_sbj, l-1, 1:3) = ...
                        all_attn_logits(m, a, cnt_sbj, l, curr_attn_I) ...
                        - all_attn_logits(m, a, cnt_sbj, l-1, curr_attn_I);
                    [~, curr_attn_I] = sort(squeeze(all_attns{m, a}(attn_where,[1 2 3],l)), 'descend');
                    all_attn_logits_diff(m, a, cnt_sbj, l-1, 4:6) = ...
                        all_attn_logits(m, a, cnt_sbj, l, curr_attn_I) ...
                        - all_attn_logits(m, a, cnt_sbj, l-1, curr_attn_I);
                end
            end
            all_attn_logits_diff_rew(m, a, cnt_sbj) = ...
                mean(all_attn_logits_diff(m, a, cnt_sbj, curr_subj_R==1, :));
            all_attn_logits_diff_unr(m, a, cnt_sbj) = ...
                mean(all_attn_logits_diff(m, a, cnt_sbj, curr_subj_R==0, :));
        end
    end
end

%% model ablation to force attention on informative feature/conj pair

%% differential response focus more informative conjunction
%% correlate differential signal with attention weights


%% **********
%% calculate value difference in attention-less model to see interaction between learning and attention
%% difference in parameters between attention and non-attention models (especially diff in lrs)
%% **********

%% **********
%% consistency of behavior with simulated models with attention to see if it accounts for individual differences
%% **********



%%

