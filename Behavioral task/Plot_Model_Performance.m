clc
clear
close all
addpath("PRLexp/inputs/")
addpath("PRLexp/SubjectData/")
addpath("files")
%% load result files
attns = load('./files/RPL2Analysis_Attention.mat') ;
feat = load('./files/RPL2Analysisv3_5_FeatureBased') ;
obj = load('./files/RPL2Analysisv3_5_FeatureObjectBased') ;
conj  = load('./files/RPL2Analysisv3_5_ConjunctionBased') ;

subjects = {...
    'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', ...
    'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', ...
    'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', ...
    'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', ...
    'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', ...
    'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'CC', 'DD', ...
    'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', ...
    'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', ...
    'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ'} ;

attn_modes = ["const", "diff", "sum", "max"];
attn_modes_choice = ["const", "diff", "sum", "max"];
attn_modes_learn = [" X const", " X diff", " X sum", " X max"];

[attn_mode_choice, attn_mode_learn] = meshgrid(attn_modes_choice, attn_modes_learn);
xlabels = strcat(attn_mode_choice(:), attn_mode_learn(:));
xlabels = reshape(reshape(xlabels, [4 4])', [16, 1]);
disp(xlabels);

ntrials = 432;
ntrialPerf       = 33:432;
perfTH           = 0.5 + 2*sqrt(.5*.5/length(ntrialPerf)) ;

cmap = lines(256);

for cnt_sbj = 1:length(subjects)
    disp(cnt_sbj)
    inputname   = ['./PRLexp/inputs/input_', subjects{cnt_sbj} , '.mat'] ;
    resultsname = ['./PRLexp/SubjectData/PRL_', subjects{cnt_sbj} , '.mat'] ;
    
    load(inputname)
    load(resultsname)
    
    rew{cnt_sbj}                  = results.reward ;
    [~, idxMax]                   = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew{cnt_sbj}            = results.choice' == idxMax ;
    perfMean(cnt_sbj)             = nanmean(choiceRew{cnt_sbj}(ntrialPerf)) ;
end
idxperf = find(perfMean>perfTH);
% idxperf = 1:length(subjects);

%% load results with attn

for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            w(i1, i2, cnt_sbj) = attns.mlparRL2conj_decay_attn_constr{i1, i2, idxperf(cnt_sbj)}(100);
        end
    end
end
wAIC = nan*w;
wAIC(1,1,:) = 2.*w(1,1,:)+2*8;
wAIC(1,2:end,:) = 2.*w(1,2:end,:)+2*10;
wAIC(2:end,1,:) = 2.*w(2:end,1,:)+2*10;
wAIC(2:end,2:end,:) = 2.*w(2:end,2:end,:)+2*12;

wBIC = nan*w;
wBIC(1,1,:) = 2.*w(1,1,:)+8*log(ntrials);
wBIC(1,2:end,:) = 2.*w(1,2:end,:)+10*log(ntrials);
wBIC(2:end,1,:) = 2.*w(2:end,1,:)+10*log(ntrials);
wBIC(2:end,2:end,:) = 2.*w(2:end,2:end,:)+12*log(ntrials);

figure;
mw = mean(w,3);
mwAIC = mean(wAIC, 3);
mwBIC = mean(wBIC, 3);
sw = std(w,1,3);
swAIC = std(wAIC,1,3);
swBIC = std(wBIC,1,3);
[~,I0] = sort(mwAIC(:));
bar(mwBIC(I0), 'FaceColor', cmap(3,:));hold on;
bar(mwAIC(I0), 'FaceColor', cmap(2,:));
bar(2*mw(I0), 'FaceColor', cmap(1,:));
ylim([400, 540]);
xticks(1:16)
xticklabels(xlabels(I0))
hold on;
er = errorbar(2*mw(I0),sw(I0)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(mwAIC(I0),swAIC(I0)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(mwBIC(I0),swBIC(I0)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
title('Feature+Conjunction Constrained Goodness-of-Fit')



for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            x(i1, i2, cnt_sbj) = attns.mlparRL2conj_decay_attn{i1, i2, idxperf(cnt_sbj)}(100);
        end
    end
end
xAIC = nan*x;
xAIC(1,1,:) = 2.*x(1,1,:)+2*8;
xAIC(1,2:end,:) = 2.*x(1,2:end,:)+2*10;
xAIC(2:end,1,:) = 2.*x(2:end,1,:)+2*10;
xAIC(2:end,2:end,:) = 2.*x(2:end,2:end,:)+2*12;

xBIC = nan*x;
xBIC(1,1,:) = 2.*x(1,1,:)+8*log(ntrials);
xBIC(1,2:end,:) = 2.*x(1,2:end,:)+10*log(ntrials);
xBIC(2:end,1,:) = 2.*x(2:end,1,:)+10*log(ntrials);
xBIC(2:end,2:end,:) = 2.*x(2:end,2:end,:)+12*log(ntrials);

figure;
mx = mean(x,3);
mxAIC = mean(xAIC, 3);
mxBIC = mean(xBIC, 3);
sx = std(x,1,3);
sxAIC = std(xAIC,1,3);
sxBIC = std(xBIC,1,3);
[~,I1] = sort(mxAIC(:));
bar(mxBIC(I1), 'FaceColor', cmap(3,:));hold on;
bar(mxAIC(I1), 'FaceColor', cmap(2,:));
bar(2*mx(I1), 'FaceColor', cmap(1,:));
ylim([400, 540]);
xticks(1:16)
xticklabels(xlabels(I1))
hold on;
er = errorbar(2*mx(I1),sx(I1)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(mxAIC(I1),sxAIC(I1)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(mxBIC(I1),sxBIC(I1)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
title('Feature+Conjunction Goodness-of-Fit')

for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            y(i1, i2, cnt_sbj) = attns.mlparRL2ftobj_decay_attn{i1, i2, idxperf(cnt_sbj)}(100);
        end
    end
end

yAIC = nan*y;
yAIC(1,1,:) = 2.*y(1,1,:)+2*8;
yAIC(1,2:end,:) = 2.*y(1,2:end,:)+2*9;
yAIC(2:end,1,:) = 2.*y(2:end,1,:)+2*9;
yAIC(2:end,2:end,:) = 2.*y(2:end,2:end,:)+2*10;

yBIC = nan*y;
yBIC(1,1,:) = 2.*y(1,1,:)+8*log(ntrials);
yBIC(1,2:end,:) = 2.*y(1,2:end,:)+9*log(ntrials);
yBIC(2:end,1,:) = 2.*y(2:end,1,:)+9*log(ntrials);
yBIC(2:end,2:end,:) = 2.*y(2:end,2:end,:)+10*log(ntrials);


figure;
my = mean(y,3);
myAIC = mean(yAIC, 3);
myBIC = mean(yBIC, 3);
sy = std(y,1,3);
syAIC = std(yAIC,1,3);
syBIC = std(yBIC,1,3);
[~,I2] = sort(myAIC(:));
bar(myBIC(I2), 'FaceColor', cmap(3,:));hold on;
bar(myAIC(I2), 'FaceColor', cmap(2,:));
bar(2*my(I2), 'FaceColor', cmap(1,:));
ylim([400, 540]);
xticks(1:16)
xticklabels(xlabels(I2))
hold on;
er = errorbar(2*my(I2),sy(I2)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(myAIC(I2),syAIC(I2)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(myBIC(I2),syBIC(I2)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
title('Feature+Object Goodness-of-Fit')

for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            z(i1, i2, cnt_sbj) = attns.mlparRL2ft_decay_attn{i1, i2, idxperf(cnt_sbj)}(100);
        end
    end
end

zAIC = nan*z;
zAIC(1,1,:) = 2.*z(1,1,:)+2*5;
zAIC(1,2:end,:) = 2.*z(1,2:end,:)+2*6;
zAIC(2:end,1,:) = 2.*z(2:end,1,:)+2*6;
zAIC(2:end,2:end,:) = 2.*z(2:end,2:end,:)+2*7;

zBIC = nan*z;
zBIC(1,1,:) = 2.*z(1,1,:)+5*log(ntrials);
zBIC(1,2:end,:) = 2.*z(1,2:end,:)+6*log(ntrials);
zBIC(2:end,1,:) = 2.*z(2:end,1,:)+6*log(ntrials);
zBIC(2:end,2:end,:) = 2.*z(2:end,2:end,:)+7*log(ntrials);


figure;
mz = mean(z,3);
mzAIC = mean(zAIC, 3);
mzBIC = mean(zBIC, 3);
sz = std(z,1,3);
szAIC = std(zAIC,1,3);
szBIC = std(zBIC,1,3);
[~,I3] = sort(mzAIC(:));
bar(mzBIC(I3), 'FaceColor', cmap(3,:));hold on;
bar(mzAIC(I3), 'FaceColor', cmap(2,:));
bar(2*mz(I3), 'FaceColor', cmap(1,:));
ylim([400, 540]);
xticks(1:16)
xticklabels(xlabels(I3))
hold on;
er = errorbar(2*mz(I3),sz(I3)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(mzAIC(I3),szAIC(I3)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
er = errorbar(mzBIC(I3),szBIC(I3)/sqrt(ntrials));
er.Color = [0 0 0];
er.LineStyle = 'none';
title('Feature Goodness-of-Fit')

I = [I0 I1 I2 I3];

%% load results without attn and compare

% TODO: do best model on both sides

% for i1=1:3
%     for cnt_sbj=1:length(idxperf)
%         x0(i1, cnt_sbj) = conj.mlparRL2conj_decay{i1, idxperf(cnt_sbj)}(100);
%     end
% end
% disp(strcat("p-val for differnce between conj model w/ and w/o attn: ", string(ranksum(x(:), x0(:)))))
% 
% figure;
% histogram(x(:), 'Normalization', 'probability', 'BinEdges', 80:10:280);
% hold on;
% histogram(x0(:), 'Normalization', 'probability', 'BinEdges', 80:10:280);
% legend('w/ attn', 'w/o attn', 'Location', 'northwest');
% title('Feature+Conj Goodness-of-fit')
% 
% for i1=1:3
%     for cnt_sbj=1:length(idxperf)
%         y0(i1, cnt_sbj) = obj.mlparRL2ftobj_decay{i1, idxperf(cnt_sbj)}(100);
%     end
% end
% disp(strcat("p-val for differnce between ft+obj model w/ and w/o attn: ", string(ranksum(y(:), y0(:)))))
% 
% figure;
% histogram(y(:), 'Normalization', 'probability', 'BinEdges', 80:10:280);
% hold on;
% histogram(y0(:), 'Normalization', 'probability', 'BinEdges', 80:10:280);
% legend('w/ attn', 'w/o attn', 'Location', 'northwest');
% title('Feature+Obj Goodness-of-fit')
% 
% for cnt_sbj=1:length(idxperf)
%     z0(cnt_sbj) = feat.mlparRL2_decay{idxperf(cnt_sbj)}(100);
% end
% disp(strcat("p-val for differnce between ft model w/ and w/o attn: ", string(ranksum(z(:), z0(:)))))
% 
% figure;
% histogram(z(:), 'Normalization', 'probability', 'BinEdges', 80:10:280);
% hold on;
% histogram(z0(:), 'Normalization', 'probability', 'BinEdges', 80:10:280);
% legend('w/ attn', 'w/o attn', 'Location', 'northwest');
% title('Feature Goodness-of-fit')
%% compare different types of attn
for i1=1:4
    for i2=1:4
        for j1=1:4
            for j2=1:4
                curr_ind_xx = sub2ind([4 4], i1, i2);
                curr_ind_yy = sub2ind([4 4], j1, j2);
                
                [p,h,stats] = signrank(squeeze(wAIC(i1,i2,:)), squeeze(wAIC(j1,j2,:)));
                attn_no_attn_ps(1, curr_ind_xx,curr_ind_yy) = p;
                attn_no_attn_diffs(1, curr_ind_xx,curr_ind_yy) = mean(wAIC(i1,i2,:))-mean(wAIC(j1,j2,:));
                
                [p,h,stats] = signrank(squeeze(xAIC(i1,i2,:)), squeeze(xAIC(j1,j2,:)));
                attn_no_attn_ps(2, curr_ind_xx,curr_ind_yy) = p;
                attn_no_attn_diffs(2, curr_ind_xx,curr_ind_yy) = mean(xAIC(i1,i2,:))-mean(xAIC(j1,j2,:));
                
                [p,h,stats] = signrank(squeeze(yAIC(i1,i2,:)), squeeze(yAIC(j1,j2,:)));
                attn_no_attn_ps(3, curr_ind_xx,curr_ind_yy) = p;
                attn_no_attn_diffs(3, curr_ind_xx,curr_ind_yy) = mean(yAIC(i1,i2,:))-mean(yAIC(j1,j2,:));
                
                [p,h,stats] = signrank(squeeze(zAIC(i1,i2,:)), squeeze(zAIC(j1,j2,:)));
                attn_no_attn_ps(4, curr_ind_xx,curr_ind_yy) = p;
                attn_no_attn_diffs(4, curr_ind_xx,curr_ind_yy) = mean(zAIC(i1,i2,:))-mean(zAIC(j1,j2,:));
            end
        end
    end
end

disp(strcat("Number of significantly different pairs: ", string(sum(attn_no_attn_ps(:)<0.05))))

attn_no_attn_stars = sig2ast(attn_no_attn_ps);

plot_titles = {"Feature+Conj Constrained Model", "Feature+Conj Model", "Feature+Obj Model", "Feature Model"};

for i=1:4
    figure;
    imagesc(squeeze(attn_no_attn_diffs(i,I(:,i),I(:,i))));
    [txs, tys] = meshgrid(1:16, 1:16);
    text(txs(:)-0.2, tys(:), attn_no_attn_stars(i,I(:,i),I(:,i)), 'Color', [150,150,150]/255, 'FontSize',7);
    colorbar();
    colormap bluewhitered;
    xticks(1:16)
    yticks(1:16)
    xticklabels(xlabels(I(:,i)));
    yticklabels(xlabels(I(:,i)));
    title(plot_titles{i});
    pbaspect([1 1 1])
end

%% load ML params

for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            wpar(i1, i2, cnt_sbj, :) = attns.mlparRL2conj_decay_attn_constr{i1, i2, idxperf(cnt_sbj)}(1:12);
        end
    end
end


for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            xpar(i1, i2, cnt_sbj, :) = attns.mlparRL2conj_decay_attn{i1, i2, idxperf(cnt_sbj)}(1:12);
        end
    end
end

for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            ypar(i1, i2, cnt_sbj, :) = attns.mlparRL2ftobj_decay_attn{i1, i2, idxperf(cnt_sbj)}(1:10);
        end
    end
end

for i1=1:4
    for i2=1:4
        for cnt_sbj=1:length(idxperf)
            zpar(i1, i2, cnt_sbj, :) = attns.mlparRL2ft_decay_attn{i1, i2, idxperf(cnt_sbj)}(1:7);
        end
    end
end

%% Simulate model with best param


subjects = {...
    'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', ...
    'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', ...
    'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', ...
    'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', ...
    'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', ...
    'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'CC', 'DD', ...
    'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', ...
    'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', ...
    'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ'} ;


for cnt_sbj=1:length(idxperf)
    disp(['Subject: ', num2str(idxperf(cnt_sbj))])
    inputname   = ['./PRLexp/inputs/input_', lower(subjects{idxperf(cnt_sbj)}) , '.mat'] ;
    resultsname = ['./PRLexp/SubjectData/PRL_', lower(subjects{idxperf(cnt_sbj)}) , '.mat'] ;

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

    for i1=1:4
        for i2=1:4
%             disp(['Attention For Choice: ', attn_modes(i1),', Learning: ', attn_modes(i2)])
            sessdata = struct();
            sesdata.flagUnr               = 1 ;
            sesdata.sig                   = 0.2 ;
            sesdata.input                 = input ;
            sesdata.expr                  = expr ;
            sesdata.results               = results ;

            % F
            sesdata.flag_couple = 0 ;
            sesdata.flag_updatesim = 0 ;
            sesdata.flagSepAttn = 1;
            sesdata.attn_mode_choice = attn_modes(i1);
            sesdata.attn_mode_learn = attn_modes(i2);
            sesdata.NparamBasic = 3 ;
            if sesdata.flagUnr==1
                sesdata.Nalpha = 2 ;
            else
                sesdata.Nalpha = 1 ;
            end
            sesdata.Nbeta = 2;

            [lls, vs, sim_attns] = fMLchoiceLL_RL2ftdecayattn(zpar(i1, i2, cnt_sbj,:), sesdata);
            LL_ft(i1, i2, cnt_sbj, :)= lls;
            Vs_ft(i1, i2, cnt_sbj, :, :) = vs;
            As_ft(i1, i2, cnt_sbj, :, :, :) = sim_attns;

            % F+O
            sesdata.flag_couple = 0 ;
            sesdata.NparamBasic = 4 ;
            sesdata.flatSepAttn = 1;
            if sesdata.flagUnr==1
                sesdata.Nalpha = 4 ;
            else
                sesdata.Nalpha = 2 ;
            end
            sesdata.Nbeta = 2;
            sesdata.attn_mode_choice = attn_modes(i1);
            sesdata.attn_mode_learn = attn_modes(i2);

            [lls, vs, sim_attns] = fMLchoiceLL_RL2ftobjdecayattn(ypar(i1, i2, cnt_sbj,:), sesdata);
            LL_ftobj(i1, i2, cnt_sbj, :)= lls;
            Vs_ftobj(i1, i2, cnt_sbj, :, :) = vs;
            As_ftobj(i1, i2, cnt_sbj, :, :, :) = sim_attns;


            % F+C
            sesdata.flag_couple = 0 ;
            NparamBasic = 4 ;
            sesdata.flatSepAttn = 1;
            if sesdata.flagUnr==1
                sesdata.Nalpha = 4 ;
            else
                sesdata.Nalpha = 2 ;
            end
            sesdata.Nbeta = 4;
            sesdata.attn_mode_choice = attn_modes(i1);
            sesdata.attn_mode_learn = attn_modes(i2);

            [lls, vs, sim_attns] = fMLchoiceLL_RL2conjdecayattn(xpar(i1, i2, cnt_sbj,:), sesdata);
            LL_ftconj(i1, i2, cnt_sbj, :)= lls;
            Vs_ftconj(i1, i2, cnt_sbj, :, :) = vs;
            As_ftconj(i1, i2, cnt_sbj, :, :, :) = sim_attns;

            % F+C constr
            sesdata.flag_couple = 0 ;
            NparamBasic = 4 ;
            sesdata.flatSepAttn = 1;
            if sesdata.flagUnr==1
                sesdata.Nalpha = 4 ;
            else
                sesdata.Nalpha = 2 ;
            end
            sesdata.Nbeta = 4;
            sesdata.attn_mode_choice = attn_modes(i1);
            sesdata.attn_mode_learn = attn_modes(i2);

            [lls, vs, sim_attns] = fMLchoiceLL_RL2conjdecayattn_constrained(wpar(i1, i2, cnt_sbj,:), sesdata);
            LL_ftconj_constr(i1, i2, cnt_sbj, :)= lls;
            Vs_ftconj_constr(i1, i2, cnt_sbj, :, :) = vs;
            As_ftconj_constr(i1, i2, cnt_sbj, :, :, :) = sim_attns;
        end
    end 
end

%% Plot Attention

wSize = 1;

attn_mode_plot_1 = 3;
attn_mode_plot_2 = 4;

figure;
subplot(121)
pm1 = plot(movmean(squeeze(mean(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:),3))', wSize));
title('F+C Tied Attention Weight For Choice');
xlim([0 433])
subplot(122)
pm2 = plot(movmean(squeeze(mean(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:),3))', wSize));
title('F+C Tied Attention Weight For Learning');
for i=1:3
    xxs = [1:ntrials, ntrials:-1:1];
    subplot(121)
    yys = [movmean(squeeze(mean(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)+movmean(squeeze(std(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)-movmean(squeeze(std(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    subplot(121)
    p1=patch(xxs, yys, cmap(i,:));hold on;
    p1.FaceAlpha = 0.2;
    p1.EdgeAlpha = 0.;
    subplot(122)
    yys = [movmean(squeeze(mean(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)+movmean(squeeze(std(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)-movmean(squeeze(std(As_ftconj_constr(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    p2=patch(xxs, yys, cmap(i,:));hold on;
    p2.FaceAlpha = 0.2;
    p2.EdgeAlpha = 0.;
end
legend(pm1, {'shape+colorXpattern', 'color+shapeXpattern', 'pattern+colorXshape'}, 'Location', 'southwest');
legend(pm2, {'shape+colorXpattern', 'color+shapeXpattern', 'pattern+colorXshape'}, 'Location', 'southwest');

attn_mode_plot_1 = 1;
attn_mode_plot_2 = 3;

figure;
subplot(121)
pm1 = plot(movmean(squeeze(mean(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:),3))', wSize));
title('F Attention Weight For Choice');
xlim([0 433])
subplot(122)
pm2 = plot(movmean(squeeze(mean(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:),3))', wSize));
title('F Attention Weight For Learning');
xlim([0 433])
for i=1:3
    xxs = [1:ntrials, ntrials:-1:1];
    yys = [movmean(squeeze(mean(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)+movmean(squeeze(std(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)-movmean(squeeze(std(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    subplot(121)
    p1=patch(xxs, yys, cmap(i,:));hold on;
    p1.FaceAlpha = 0.2;
    p1.EdgeAlpha = 0.;
    subplot(122)
    yys = [movmean(squeeze(mean(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)+movmean(squeeze(std(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)-movmean(squeeze(std(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    p2=patch(xxs, yys, cmap(i,:));hold on;
    p2.FaceAlpha = 0.2;
    p2.EdgeAlpha = 0.;
end
legend(pm1, {'shape', 'color', 'pattern'}, 'Location', 'southwest');
legend(pm2, {'shape', 'color', 'pattern'}, 'Location', 'southwest');


attn_mode_plot_1 = 2;
attn_mode_plot_2 = 3;

figure;
subplot(121)
pm1 = plot(movmean(squeeze(mean(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:),3))', wSize));
title('F+O Attention Weight For Choice');
xlim([0 433])
subplot(122)
pm2 = plot(movmean(squeeze(mean(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:),3))', wSize));
title('F+O Attention Weight For Learning');
xlim([0 433])
for i=1:3
    xxs = [1:ntrials, ntrials:-1:1];
    yys = [movmean(squeeze(mean(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)+movmean(squeeze(std(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)-movmean(squeeze(std(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    subplot(121)
    p1=patch(xxs, yys, cmap(i,:));hold on;
    p1.FaceAlpha = 0.2;
    p1.EdgeAlpha = 0.;
    subplot(122)
    yys = [movmean(squeeze(mean(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)+movmean(squeeze(std(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)-movmean(squeeze(std(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    p2=patch(xxs, yys, cmap(i,:));hold on;
    p2.FaceAlpha = 0.2;
    p2.EdgeAlpha = 0.;
end
legend(pm1, {'shape', 'color', 'pattern'}, 'Location', 'southwest');
legend(pm2, {'shape', 'color', 'pattern'}, 'Location', 'southwest');


attn_mode_plot_1 = 3;
attn_mode_plot_2 = 3;

figure;
subplot(121)
pm1 = plot(movmean(squeeze(mean(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:),3))', wSize));
title('F+C Separate Attention Weight For Choice');
xlim([0 433])
subplot(122)
pm2 = plot(movmean(squeeze(mean(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:),3))', wSize));
title('F+C Separate Attention Weight For Learning');
for i=1:6
    xxs = [1:ntrials, ntrials:-1:1];
    subplot(121)
    yys = [movmean(squeeze(mean(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)+movmean(squeeze(std(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),3))', wSize)-movmean(squeeze(std(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    subplot(121)
    p1=patch(xxs, yys, cmap(i,:));hold on;
    p1.FaceAlpha = 0.2;
    p1.EdgeAlpha = 0.;
    subplot(122)
    yys = [movmean(squeeze(mean(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)+movmean(squeeze(std(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)), ...
           fliplr(movmean(squeeze(mean(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),3))', wSize)-movmean(squeeze(std(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,i,:),1,3))', wSize)/sqrt(length(idxperf)))];
    p2=patch(xxs, yys, cmap(i,:));hold on;
    p2.FaceAlpha = 0.2;
    p2.EdgeAlpha = 0.;
end
legend(pm1, {'shape', 'color', 'pattern', 'patternXshape', 'patternXcolor', 'shapeXcolor'}, 'Location', 'southwest');
legend(pm2, {'shape', 'color', 'pattern', 'patternXshape', 'patternXcolor', 'shapeXcolor'}, 'Location', 'southwest');

%% Get entropic measures
% ce1 = squeeze(cross_entropy(squeeze(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:)), repmat([0 1 0], length(idxperf), 1, ntrials)));
% ce2 = squeeze(cross_entropy(squeeze(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:)), repmat([0 1 0], length(idxperf), 1, ntrials)));
% ent1 = squeeze(entropy(squeeze(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:))));
% ent2 = squeeze(entropy(squeeze(As_ft(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:))));
% 
% figure
% subplot(121)
% plot(mean(ce1, 1));
% hold on;
% plot(mean(ce2, 1));
% title('Cross Entropy')
% legend({"Choice", 'Learning'})
% subplot(122)
% plot(mean(ent1, 1));
% hold on;
% plot(mean(ent2, 1));
% title('Entropy')
% legend({"Choice", 'Learning'})
% sgtitle('Feature Model')
% 
% 
% ce1 = squeeze(cross_entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:)), repmat([0 1 0], length(idxperf), 1, ntrials)));
% ce2 = squeeze(cross_entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:)), repmat([0 1 0], length(idxperf), 1, ntrials)));
% ent1 = squeeze(entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,:,:))));
% ent2 = squeeze(entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,:,:))));
% 
% figure
% subplot(121)
% plot(mean(ce1, 1));
% hold on;
% plot(mean(ce2, 1));
% title('Cross Entropy')
% legend({"Choice", 'Learning'})
% subplot(122)
% plot(mean(ent1, 1));
% hold on;
% plot(mean(ent2, 1));
% title('Entropy')
% legend({"Choice", 'Learning'})
% sgtitle('Feature+Obj Model')
% 
% ce1ft = squeeze(cross_entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,1:3,:)), repmat([0 1 0], length(idxperf), 1, ntrials)));
% ce1conj = squeeze(cross_entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,4:6,:)), repmat([1 0 0], length(idxperf), 1, ntrials)));
% ce2ft = squeeze(cross_entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,1:3,:)), repmat([0 1 0], length(idxperf), 1, ntrials)));
% ce2conj = squeeze(cross_entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,4:6,:)), repmat([1 0 0], length(idxperf), 1, ntrials)));
% ent1ft = squeeze(entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,1:3,:))));
% ent1conj = squeeze(entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,1,4:6,:))));
% ent2ft = squeeze(entropy(squeeze(As_ftconj(attn_mode_plot_1,attn_mode_plot_2,:,2,1:3,:))));
% ent2conj = squeeze(entropy(squeeze(As_ftobj(attn_mode_plot_1,attn_mode_plot_2,:,2,4:6,:))));
% 
% figure
% subplot(121)
% plot(mean(ce1ft, 1));
% hold on;
% plot(mean(ce1conj, 1));
% plot(mean(ce2ft, 1));
% plot(mean(ce2conj, 1));
% title('Cross Entropy')
% legend({"Choice, Feature", "Choice, Conj", 'Learning, Feature', "Learning, Conj"})
% subplot(122)
% plot(mean(ent1ft, 1));
% hold on;
% plot(mean(ent1conj, 1));
% plot(mean(ent2ft, 1));
% plot(mean(ent2conj, 1));
% title('Entropy')
% legend({"Choice, Feature", "Choice, Conj", 'Learning, Feature', "Learning, Conj"})
% sgtitle('Feature+Conj Model')

%% make results table

num_digs_round = 1;

restbl_w = table(xlabels(I0), strcat(string(round(mw(I0), num_digs_round)), char(177), string(round(sw(I0)/sqrt(ntrials),num_digs_round))), ...
               strcat(string(round(mwAIC(I0),num_digs_round)), char(177), string(round(swAIC(I0)/sqrt(ntrials),num_digs_round))), ...
               strcat(string(round(mwBIC(I0),num_digs_round)), char(177), string(round(swBIC(I0)/sqrt(ntrials),num_digs_round))), ...
               'VariableNames', {'Attn Type', '-LL', 'AIC', 'BIC'});
restbl_x = table(xlabels(I1), strcat(string(round(mx(I1), num_digs_round)), char(177), string(round(sx(I1)/sqrt(ntrials), num_digs_round))), ...
               strcat(string(round(mxAIC(I1), num_digs_round)), char(177), string(round(sxAIC(I1)/sqrt(ntrials), num_digs_round))), ...
               strcat(string(round(mxBIC(I1), num_digs_round)), char(177), string(round(sxBIC(I1)/sqrt(ntrials), num_digs_round))), ...
               'VariableNames', {'Attn Type', '-LL', 'AIC', 'BIC'});
restbl_y = table(xlabels(I2), strcat(string(round(my(I2), num_digs_round)), char(177), string(round(sy(I2)/sqrt(ntrials), num_digs_round))), ...
               strcat(string(round(myAIC(I2), num_digs_round)), char(177), string(round(syAIC(I2)/sqrt(ntrials), num_digs_round))), ...
               strcat(string(round(myBIC(I2), num_digs_round)), char(177), string(round(syBIC(I2)/sqrt(ntrials), num_digs_round))), ...
               'VariableNames', {'Attn Type', '-LL', 'AIC', 'BIC'});
restbl_z = table(xlabels(I3), strcat(string(round(mz(I3), num_digs_round)), char(177), string(round(sz(I3)/sqrt(ntrials), num_digs_round))), ...
               strcat(string(round(mzAIC(I3), num_digs_round)), char(177), string(round(szAIC(I3)/sqrt(ntrials), num_digs_round))), ...
               strcat(string(round(mzBIC(I3), num_digs_round)), char(177), string(round(szBIC(I3)/sqrt(ntrials), num_digs_round))), ...
               'VariableNames', {'Attn Type', '-LL', 'AIC', 'BIC'});

%% get best model among each input
[~, II] = max(-wAIC, [], [1 2], 'linear');
[II1, II2, ~] = ind2sub([4 4 41], II);
II = sub2ind([4 4], II1, II2);
figure;
histogram(II, 'BinEdges', 1:17);
xticks((1:16)+0.5);
xticklabels(xlabels);
title('F+C Tied')


[~, II] = max(-xAIC, [], [1 2], 'linear');
[II1, II2, ~] = ind2sub([4 4 41], II);
II = sub2ind([4 4], II1, II2);
figure;
histogram(II, 'BinEdges', 1:17);
xticks((1:16)+0.5);
xticklabels(xlabels);
title('F+C Untied')


[~, II] = max(-yAIC, [], [1 2], 'linear');
[II1, II2, ~] = ind2sub([4 4 41], II);
II = sub2ind([4 4], II1, II2);
figure;
histogram(II, 'BinEdges', 1:17);
xticks((1:16)+0.5);
xticklabels(xlabels);
title('F+O')


[~, II] = max(-zAIC, [], [1 2], 'linear');
[II1, II2, ~] = ind2sub([4 4 41], II);
II = sub2ind([4 4], II1, II2);
figure;
histogram(II, 'BinEdges', 1:17);
xticks((1:16)+0.5);
xticklabels(xlabels);
title('F')

%%
figure
b = bar([mean(reshape(wAIC, [16, 41]),2)';mean(reshape(xAIC, [16, 41]),2)';mean(reshape(yAIC, [16, 41]),2)';mean(reshape(zAIC, [16, 41]),2)'], 'stacked', 'FaceColor','flat');
for k = 1:16
b(k).CData = k;
end
colormap jet
xlim([0 6])
legend(b, xlabels)
xticklabels({'F+C Tied', 'F+C Untied', 'F+O', 'F'})
