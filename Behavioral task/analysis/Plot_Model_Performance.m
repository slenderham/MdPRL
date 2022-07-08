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

set(0,'defaultAxesFontSize',15)
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
idxperf = find(idxperf);
% idxperf = 1:length(subjects);

figure
plot_shaded_errorbar(mean(movmean(rew(idxperf,:), 20, 2))', std(movmean(rew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, 'k');
plot_shaded_errorbar(mean(movmean(choiceRew(idxperf,:), 20, 2))', std(movmean(choiceRew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, [0.5 0.5 0.5]);
xlabel('Trial Number')
ylabel('Performance')
legend({'', 'Reward', '', 'Proportion Better'})
xlim([0, 432])


%% load results with attn and ML params

attns = load('../files/RPL2Analysis_Attention_merged_rep50.mat') ;

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            lls(m, a, cnt_sbj) = attns.fit_results{m, a, idxperf(cnt_sbj)}.fval;            
            AICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+2*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
            BICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+log(ntrials)*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
        end
    end
end

%% bayesian model selection

[alpha_AIC,exp_r_AIC,xp_AIC,pxp_AIC,bor_AIC,g_AIC] = bms(reshape(-permute(AICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_AIC);
figure;
h = bar(reshape(pxp_AIC, 10, 5));hold on;

hx_pos = nan(10, 5);
for i = 1:5
    hx_pos(:,i) = h(i).XEndPoints;
end

plot(hx_pos(:), alpha_AIC/sum(alpha_AIC), 'ko', 'DisplayName', "E[Freq]")
xticks(1:10)
xticklabels(attn_modes_legend)
set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
ylabel('pxp')
legend()


[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_BIC);
figure;
h = bar(reshape(pxp_BIC, 10, 5));hold on;

hx_pos = nan(10, 5);
for i = 1:5
    hx_pos(:,i) = h(i).XEndPoints;
end

plot(hx_pos(:), alpha_BIC/sum(alpha_BIC), 'ko', 'DisplayName', "E[Freq]")
xticks(1:10)
xticklabels(attn_modes_legend)
set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
ylabel('pxp')
legend()

[~, best_model_inds] = max(g_BIC);
% title('Posterior Model Probability')

[alpha_input,exp_r_input,xp_input,pxp_input,bor_input,g_input] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell(reshape(1:50, [10, 5])', repmat([1], 1, 5)));
disp(bor_input);
figure;
h = bar(pxp_input);hold on;
plot(alpha_input/sum(alpha_input), 'ko', 'DisplayName', "E[Freq]")
xticks(1:5)
xticklabels({'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'})
ylabel('pxp')

[alpha_attn,exp_r_attn,xp_attn,pxp_attn,bor_attn,g_attn] = bms(reshape(-BICs/2, [50, length(idxperf)])', ...
                                mat2cell(reshape(1:50, [5, 10])', repmat([1], 1, 10)));
disp(bor_attn);
figure;
h = bar(pxp_attn);hold on;
plot(alpha_attn/sum(alpha_attn), 'ko', 'DisplayName', "E[Freq]")
xticks(1:10)
xticklabels(attn_modes_legend(:,1))
ylabel('pxp')

%%
t = tiledlayout(5, 7, 'TileSpacing','compact');
nexttile([1 6])
% imagesc(alpha_attn'/sum(alpha_attn));
imagesc(pxp_attn)
caxis([0 1])
xticks([])
yticks([])
nexttile
axis off
nexttile([4 6])
imagesc(reshape(alpha_BIC/sum(alpha_BIC), 10, 5)');
imagesc(reshape(pxp_BIC, 10, 5)');
caxis([0 1])
xticks(1:10)
xticklabels(attn_modes_legend)
yticks(1:5)
yticklabels({'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
nexttile([4 1])
% imagesc(alpha_input/sum(alpha_input));
imagesc(pxp_input')
caxis([0 1])
xticks([])
yticks([])
colormap(flipud(bone))
cb = colorbar;
cb.Layout.Tile = 'South';

%% Simulate model with best param

for m = 1:length(all_model_names)
    disp("=======================================================");
    disp(strcat("Fitting model ", all_model_names(m)));
    basic_params = cell(length(subjects_inputs)); % store the attention-less model's parameters for each model type
    for a = 1:length(attn_modes)
        disp("-------------------------------------------------------");
        disp(strcat("Fitting attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        parfor cnt_sbj = 1:length(subjects_inputs)
%             disp(strcat("Fitting subject ", num2str(cnt_sbj)));
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

            % load attn type (const, diff, sum, max) and attn
            % time(none, choice, learning, both)
            sesdata.attn_op = attn_modes(a,1);
            sesdata.attn_time = attn_modes(a,2);

            % load best params
            best_pars = attns.fit_results{m, a, cnt_sbj}.params;

            ll = str2func(all_model_names(m));
            
            [trial_ll, trial_values, trial_attns] = ll(best_pars, sesdata);

            all_values{m, a, cnt_sbj} = trial_values;
            all_attns{m, a, cnt_sbj} = trial_attns;
            trial_lls(m, a, cnt_sbj, :) = trial_ll;
            trial_AICs(m, a, cnt_sbj, :) = 2*trial_lls(m, a, cnt_sbj, :)+2*length(best_pars)/ntrials;
            trial_BICs(m, a, cnt_sbj, :) = 2*trial_lls(m, a, cnt_sbj, :)+log(ntrials)*length(best_pars)/ntrials;
        end
    end
end

unfilt_AICs = trial_AICs;
unfilt_BICs = trial_BICs;

trial_AICs=trial_AICs(:,:,idxperf,:);
trial_BICs=trial_BICs(:,:,idxperf,:);

%% Plot Model Evidence

wSize = 20;
clrmat = colormap('lines(5)') ;

smth_AIC = movmean(trial_AICs, [0 wSize-1], 4, 'Endpoints', 'discard');
smth_BIC = movmean(trial_BICs, [0 wSize-1], 4, 'Endpoints', 'discard');
smth_ll = movmean(trial_lls, [0 wSize-1], 4, 'Endpoints', 'discard');

figure
for i=1:5
    [~, min_attn_type] = max(alpha_BIC((i-1)*length(attn_modes)+1:i*length(attn_modes)));
%     disp(min_attn_type)
    l(i) = plot_shaded_errorbar(squeeze(median(smth_BIC(i,min_attn_type,:,:), [1 2 3])), ...
                                squeeze(1.25*std(smth_BIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(length(idxperf)), ...
                                wSize, clrmat(i,:));hold on
end
legend(l, ["F", "F+O", "F+C_{feat attn}", "F+C_{untied}", "F+C_{tied}"])
xlabel('Trial')
ylabel('Trial-wise BIC')

figure
for i=1:5
    [~, min_attn_type] = max(alpha_AIC((i-1)*length(attn_modes)+1:i*length(attn_modes)));
%     disp(min_attn_type)
    l(i) = plot_shaded_errorbar(squeeze(median(smth_AIC(i,min_attn_type,:,:), [1 2 3])), ...
                                squeeze(1.25*std(smth_AIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(length(idxperf)), ...
                                wSize, clrmat(i,:));hold on
end
legend(l, ["F", "F+O", "F+C_{feat attn}", "F+C_{untied}", "F+C_{tied}"])
xlabel('Trial')
ylabel('Trial-wise AIC')

%% compare evidence at diff epochs / detect transition

% probeTrialsAll = load(['../PRLexp/inputs_all/inputs/input_', 'aa' , '.mat']).expr.trialProbe;
% probeTrialsAll = [1 probeTrialsAll];
% 
% for i=1:length(probeTrialsAll)-1
%     binned_trial_BICs(:,:,:,i) = mean(trial_BICs(:,:,:,probeTrialsAll(i):probeTrialsAll(i+1)), 4);
% end


%% quantify effect of attention
%% sharpness of attention (entropy)? focus on the correct feature (cross entropy)

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
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
                all_model_kls(m, a, cnt_sbj, :) = squeeze(symm_kl_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,2:end), ...
                                                                      all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,1:end-1), 2));
            else
                all_model_ents(m, a, cnt_sbj, :) = squeeze((entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,:)) ...
                                                           +entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,:)))/2);
                all_model_kls(m, a, cnt_sbj, :) = squeeze((symm_kl_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,2:end), ...
                                                                       all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,1:end-1), 2) ...
                                                           +symm_kl_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,2:end), ...
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
posterior_model_ents = sum(all_model_ents.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
posterior_model_kls = sum(all_model_kls.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
plot_shaded_errorbar(squeeze(mean(posterior_model_ents, 3)), squeeze(std(posterior_model_ents, [], 3))/sqrt(length(idxperf)), 1, rgb('deepskyblue'));hold on;
plot_shaded_errorbar(squeeze(mean(posterior_model_kls, 3)), squeeze(std(posterior_model_kls, [], 3))/sqrt(length(idxperf)), 1:ntrials-1, rgb('coral'));hold on;
legend(["", "Entropy", "", "KL_{symm}"])
ylim([0, 15])
xlim([0, ntrials+10])
xlabel('Trial')

%% figure out how the flaginf is used if at all

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
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
            for d = 1:3
                if m~=3 
                    all_model_ces(m, a, cnt_sbj, :, d) = squeeze(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,d,:));
                else
                    conj_d = [2, 1, 3];
                    conj_d = conj_d(d);
                    all_model_ces(m, a, cnt_sbj, :, d) = squeeze((all_attns{m, a, idxperf(cnt_sbj)}(attn_where,d,:) ...
                                                               +all_attns{m, a, idxperf(cnt_sbj)}(attn_where,3+conj_d,:))/2);
                end
            end
        end
    end
end

% fig=  figure;
% tiledlayout(3,3);
% for a = 2:length(attn_modes)
%     nexttile;
%     for m = 1:length(all_model_names)
%         plot_shaded_errorbar(squeeze(mean(all_model_ces(m,a,:,:), 3)), squeeze(std(all_model_ces(m,a,:,:), [], 3))/sqrt(length(idxperf)), 1, clrmat(m,:));hold on;
%         ylim([0, 10])
%         if mod(a, 3)==2
%             ylabel(attn_modes(a, 1))
%         end
%         if (a-1)/3>2
%             xlabel(attn_modes(a, 2))
%         end
%     end
% end
% 
% lg  = legend(reshape([repmat([""], 5, 1) all_model_names_legend(1,:)']', 10, 1)); 
% lg.Layout.Tile = 'East';

figure
clrmat = colormap('winter(3)');
posterior_model_ces = sum(all_model_ces.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1]), [2 1 3 4 5]), [1 2]);
posterior_model_ces = squeeze(posterior_model_ces);
wSize = 1;
smth_ces = movmean(squeeze(posterior_model_ces), [0 wSize-1], 2, 'Endpoints', 'discard');
for d=[2 1 3]
    plot_shaded_errorbar(squeeze(mean(smth_ces(:,:,d), 1))', squeeze(std(smth_ces(:,:,d), [], 1))'/sqrt(length(idxperf)), wSize, clrmat(d,:));hold on;
end
legend(["", "inf", "", "noninf1", "", "noninf2"]); 
ylim([0.15, 0.6])
xlim([wSize, ntrials+10])
xlabel('Trial')
ylabel('Attention Weights')

%% extra explained variance based on diff lls
% figure;
% imagesc(1-mean(lls(5,3,:), 3)./mean(lls, 3));
% axis image
% colorbar();
% xticks(1:10)
% xticklabels(attn_modes_legend)
% yticks(1:5)
% yticklabels(all_model_names_legend')

%% differential signals in value (pairwise difference in values in different dimensions)
%% significance of "complementary" attention

channel_groups = {[0 3 6 9], ...
                  [0 3 6 9 36], ...
                  [0 3 6 9 18 27 36], ...
                  [0 3 6 9 18 27 36], ...
                  [0 3 6 9 18 27 36]};

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            for l = 1:ntrials
                cg = channel_groups{m};
                all_value_pdists(m, a, cnt_sbj, l, 1:6) = nan;
                for cg_idx = 1:length(cg)-1
                    all_value_pdists(m, a, cnt_sbj, l, cg_idx) = ...
                                 mean(pdist( ...
                                    all_values{m, a, idxperf(cnt_sbj)}(cg(cg_idx)+1:cg(cg_idx+1), l) ...
                                 ));
                end
            end
        end
    end
end

% posterior_model_pdists = nansum(all_value_pdists.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1 1]), [2 1 3 4 5]), [1 2]);
% posterior_model_pdists = squeeze(posterior_model_pdists);

figure;
clrmat=colormap('lines(6)');
clrmat = clrmat([2 1 3 4 5 6], :);
for i=[2 1 3 4 5 6]
subplot(1, 2, floor((i-1)/3)+1);
plot_shaded_errorbar(squeeze(nanmean(all_value_pdists(3:5,1,:,:,i), [1 3])), squeeze(nanstd(all_value_pdists(3:5,1,:,:,i), [], [1 3]))/sqrt(length(idxperf)), 1:ntrials, clrmat(i,:));
end
subplot(1,2,1);
ylim([0 0.2]);
yticks(0:0.025:0.2)
ylabel('Value Separability')
legend(["", "F_{inf}", "", "F_{noninf1}", "", "F_{noninf2}"]);
subplot(1,2,2);
ylim([0 0.15]);
yticks(0:0.025:0.15)
legend(["", "C_{inf}", "", "C_{noninf1}", "", "C_{noninf2}"]);
han=axes(gcf,'visible','off');
han.Title.Visible='on';
han.XLabel.Visible='on';
xlabel(han,'Trial');


figure;
clrmat=colormap('lines(6)');
clrmat = clrmat([2 1 3 4 5 6], :);
for i=[2 1 3 4 5 6]
subplot(1, 2, floor((i-1)/3)+1);
plot_shaded_errorbar(squeeze(nanmean(all_value_pdists(3,3,:,:,i), [1 3])), squeeze(nanstd(all_value_pdists(3,3,:,:,i), [], [1 3]))/sqrt(length(idxperf)), 1:ntrials, clrmat(i,:));
end
subplot(1,2,1);
ylim([0 0.2]);
yticks(0:0.025:0.2)
ylabel('Value Separability')
legend(["", "F_{inf}", "", "F_{noninf1}", "", "F_{noninf2}"]);
subplot(1,2,2);
ylim([0 0.15]);
yticks(0:0.025:0.15)
legend(["", "C_{inf}", "", "C_{noninf1}", "", "C_{noninf2}"]);
han=axes(gcf,'visible','off');
han.Title.Visible='on';
han.XLabel.Visible='on';
xlabel(han,'Trial');

figure;
clrmat=colormap('lines(6)');
clrmat = clrmat([2 1 3 4 5 6], :);
for i=[2 1 3 4 5 6]
subplot(1, 2, floor((i-1)/3)+1);
plot_shaded_errorbar(squeeze(nanmean(all_value_pdists(5,3,:,:,i), [1 3])), squeeze(nanstd(all_value_pdists(5,3,:,:,i), [], [1 3]))/sqrt(length(idxperf)), 1:ntrials, clrmat(i,:));
end
subplot(1,2,1);
ylim([0 0.2]);
yticks(0:0.025:0.2)
ylabel('Value Separability')
legend(["", "F_{inf}", "", "F_{noninf1}", "", "F_{noninf2}"]);
subplot(1,2,2);
ylim([0 0.15]);
yticks(0:0.025:0.15)
legend(["", "C_{inf}", "", "C_{noninf1}", "", "C_{noninf2}"]);
han=axes(gcf,'visible','off');
han.Title.Visible='on';
han.XLabel.Visible='on';
xlabel(han,'Trial');
%% model ablation to force attention on informative feature/conj pair

%% simulate without teacher-forcing trial to see robustness

%% run model recovery

%% look at parameter, confirmation bias?

%% differential response focus more informative conjunction
%% correlate differential signal with attention weights


%% **********
%% calculate value difference in attention-less model to see interaction between learning and attention
%% difference in parameters between attention and non-attention models (especially diff in lrs)
%% **********

%% **********
%% consistency of behavior with simulated models with attention to see if it accounts for individual differences
%% **********

all_sim_model_names = ["fMLchoiceSim_RL2ftdecayattn", ...
    "fMLchoiceSim_RL2ftobjdecayattn", ...
    "fMLchoiceSim_RL2conjdecayattn", ...
    "fMLchoiceSim_RL2conjdecayattn_onlyfattn", ...
    "fMLchoiceSim_RL2conjdecayattn_constrained"];

clear all_sim_choices all_sim_rewards all_sim_corrects all_sim_lls all_sim_values all_sim_attns
nreps = 20;
for m = [1 5]
    disp("=======================================================");
    disp(strcat("Simulating model ", all_model_names(m)));
    for a = [1 3]
        disp("-------------------------------------------------------");
        disp(strcat("Fitting attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        parfor cnt_sbj = 1:length(subjects_inputs)
            for cnt_rep = 1:nreps
                inputname   = strcat("../PRLexp/inputs_all/", subjects_inputs(cnt_sbj) , ".mat") ;
                resultsname = strcat("../PRLexp/SubjectData_all/", subjects_prl(cnt_sbj) , ".mat") ;
    
                inputs_struct = load(inputname);
                results_struct = load(resultsname);
    % 
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
    
                % load attn type (const, diff, sum, max) and attn
                % time(none, choice, learning, both)
                sesdata.attn_op = attn_modes(a,1);
                sesdata.attn_time = attn_modes(a,2);

                % load best params
                best_pars = attns.fit_results{m, a, cnt_sbj}.params;
    
                sim_model = str2func(all_sim_model_names(m));
                
                [sim_logits, sim_choices, sim_rewards, sim_values, sim_attns] = sim_model(best_pars, sesdata);

                all_sim_choices(m, a, cnt_sbj, cnt_rep, :) = sim_choices;
                all_sim_rewards(m, a, cnt_sbj, cnt_rep, :) = sim_rewards;
                [~, idxMax] = max(expr.prob{1}(input.inputTarget)) ;
                all_sim_corrects(m, a, cnt_sbj, cnt_rep, :) = sim_choices==idxMax;
                all_sim_correct_behav(m, a, cnt_sbj, cnt_rep, :) = sim_choices==results.choice';
                all_sim_lls(m, a, cnt_sbj, cnt_rep, :) = -logsigmoid(sim_logits'.*(sesdata.results.choice*2-3));
                all_sim_values{m, a, cnt_sbj, cnt_rep} = sim_values;
                all_sim_attns{m, a, cnt_sbj, cnt_rep} = sim_attns;
%                 sim_trial_BICs(m, a, cnt_sbj, :) = 2*all_sim_lls(m, a, cnt_sbj, cnt_rep)+log(ntrials)*length(best_pars)/ntrials;
            end
        end
    end
end

%%
for m = [1 5]
    for a = [1 3]
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
clrmat = colormap('winter(3)');
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

figure
plot_shaded_errorbar(squeeze(mean(movmean(all_sim_corrects(1,1,idxperf,1:nreps,:), 20, 5), [3 4])), squeeze(std(movmean(all_sim_corrects(1,1,idxperf,1:nreps,:), 20, 5), [], [3 4]))/sqrt(length(idxperf)), 1, rgb('deepskyblue'));hold on
plot_shaded_errorbar(squeeze(mean(movmean(all_sim_corrects(1,3,idxperf,1:nreps,:), 20, 5), [3 4])), squeeze(std(movmean(all_sim_corrects(1,3,idxperf,1:nreps,:), 20, 5), [], [3 4]))/sqrt(length(idxperf)), 1, rgb('blue'));hold on
plot_shaded_errorbar(squeeze(mean(movmean(all_sim_corrects(5,1,idxperf,1:nreps,:), 20, 5), [3 4])), squeeze(std(movmean(all_sim_corrects(5,1,idxperf,1:nreps,:), 20, 5), [], [3 4]))/sqrt(length(idxperf)), 1, rgb('limegreen'));hold on
plot_shaded_errorbar(squeeze(mean(movmean(all_sim_corrects(5,3,idxperf,1:nreps,:), 20, 5), [3 4])), squeeze(std(movmean(all_sim_corrects(5,3,idxperf,1:nreps,:), 20, 5), [], [3 4]))/sqrt(length(idxperf)), 1, rgb('green'));hold on
plot_shaded_errorbar(mean(movmean(choiceRew(idxperf,:), 20, 2))', std(movmean(choiceRew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, [0.5 0.5 0.5]);
xlabel('Trial')
xlim([0 ntrials+10])
ylabel('Percent Correct')
legend(["", "F_{no attn}", "", "F_{diffXL}", "", "F+C_{no attn}", "", "F+C_{tied diffXL}", "", "Human"], 'location', 'southeast')

