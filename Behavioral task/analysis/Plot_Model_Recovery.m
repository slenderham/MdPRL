clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_sim/")
addpath("../utils")
addpath("../files/")
addpath("../utils/DERIVESTsuite/DERIVESTsuite/")
addpath("../models/")
addpath("../../PRLexpv3_5v2/")

load('RPL2Analysis_Attention_model_recovery.mat')

nSample = 100;

attn_modes_choice = ["const", "diff", "sum", "max"];
attn_modes_learn = [" X const", " X diff", " X sum", " X max"];

[attn_mode_choice, attn_mode_learn] = meshgrid(attn_modes_choice, attn_modes_learn);
xlabels = strcat(attn_mode_choice(:), attn_mode_learn(:));
xlabels = reshape(reshape(xlabels, [4 4])', [16, 1]);

% disp('Calculating Performance')
%
% perfMeans = zeros(len_i_1, len_i_2, 4);
%
% ntrialPerf = 33:432;
% for cnt_samp = 1:nSamples
%     inputname   = ['../PRLexp/inputs_sim/input_', num2str(cnt_samp,'%03.f') , '.mat'] ;
%     input_struct =  load(inputname);
%     [~, idxMax] = max(input_struct.expr.prob{1}(input_struct.input.inputTarget)) ;
%
%     disp(['Subject ' num2str(cnt_samp)])
%     for i1=1:len_i_1
%         for i2=1:len_i_2
%             choice_better(cnt_samp, i1, i2, 1, :) = idxMax'==squeeze(all_ft_Cs(cnt_samp, i1, i2, 1, :));
%             choice_better(cnt_samp, i1, i2, 2, :) = idxMax'==squeeze(all_ftobj_Cs(cnt_samp, i1, i2, 1, :));
%             choice_better(cnt_samp, i1, i2, 3, :) = idxMax'==squeeze(all_ftconj_Cs(cnt_samp, i1, i2, 1, :));
%             choice_better(cnt_samp, i1, i2, 4, :) = idxMax'==squeeze(all_ftconj_constr_Cs(cnt_samp, i1, i2, 1, :));
%         end
%     end
% end
%
%
% b = bar(reshape(perfMean, 16, 4));hold on;
% nbars = 4;ngroups=16;
% x = nan(nbars, ngroups);
% for i = 1:nbars
%     x(i,:) = b(i).XEndPoints;
% end
% set(b, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{tied}'}')
% legend()
% er = errorbar(x', reshape(perfMean, 16, 4), reshape(perfStd, 16, 4)/sqrt(length(ntrialPerf*nSamples)));
% for i=1:4
%     er(i).Color = [0 0 0];
%     er(i).LineStyle = 'none';
% end

%%


sim_dim = 4;
ll_temp=0;
for i1=3:4
    for i2=3:4
        for sim_i1=3:4
            for sim_i2=3:4
                for cnt_samp=1:10
                    disp(strcat('Sample: ', num2str(cnt_samp), ...
                        ', Attn Choice: ', attn_modes{sim_i1}, ...
                        ', Attn Learn: ', attn_modes{sim_i2}));
                    inputname   = ['../PRLexp/inputs_sim/input_', num2str(cnt_samp,'%03.f'), '.mat'] ;
                
                    inputs_struct = load(inputname);
                
                    expr = inputs_struct.expr;
                    input = inputs_struct.input;
                
                    expr.shapeMap = repmat([1 2 3 ;
                        1 2 3 ;
                        1 2 3 ], 1,1,3) ;
                
                    expr.colorMap = repmat([1 1 1 ;
                        2 2 2 ;
                        3 3 3], 1,1,3) ;
                
                    expr.patternMap(:,:,1) = ones(3,3) ;
                    expr.patternMap(:,:,2) = 2*ones(3,3) ;
                    expr.patternMap(:,:,3) = 3*ones(3,3) ;

                    ll_temp(i1, i2, cnt_samp, sim_i1, sim_i2) = mlparRL2conj_decay_attn_constr{i1, i2, cnt_samp, sim_i1, sim_i2, 4}(100);

                    sesdata = struct();
                    sesdata.sig     = 1 ;
                    sesdata.input   = input ;
                    sesdata.expr    = expr ;
                    sesdata.NtrialsShort = expr.NtrialsShort ;
                    sesdata.flagUnr = 1 ;
                    sesdata.flag_couple = 0 ;
                    sesdata.flag_updatesim = 0 ;
                    sesdata.flagSepAttn = 1;

                    sesdata.attn_mode_choice = attn_modes{i1};
                    sesdata.attn_mode_learn = attn_modes{i2};
                    sesdata.results.choice = all_Cs(cnt_samp, sim_i1, sim_i2, sim_dim, :);
                    sesdata.results.reward = all_Rs(cnt_samp, sim_i1, sim_i2, sim_dim, :);
                    NparamBasic = 4 ;

                    if sesdata.flagUnr==1
                        sesdata.Nalpha = 4 ;
                    else
                        sesdata.Nalpha = 2 ;
                    end

                    if i1==1 && i2==1
                        sesdata.Nbeta = 0;
                    elseif i1==1 || i2==1
                        sesdata.Nbeta = 2;
                    else
                        sesdata.Nbeta = 4;
                    end

                    lbs = [-50,  0,  0, 0, 0, 0, 0, 0,  0,  0,  0,  0];
                    ubs = [ 50, 50, 50, 1, 1, 1, 1, 1, 50, 50, 50, 50];
                    log_priors = -log(ubs-lbs);

                    ll = @(x) sum(fMLchoiceLL_RL2conjdecayattn_constrained(x, sesdata));
                    xpar = mlparRL2conj_decay_attn_constr{i1, i2, cnt_samp, sim_i1, sim_i2, 4}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                    hess_temp{i1, i2, cnt_samp, sim_i1, sim_i2} = hessian(ll, xpar);
                    priors = sum(log_priors(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta));
                    LApprox_temp(i1, i2, cnt_samp, sim_i1, sim_i2) = ...
                        -laplace_approximation(-ll_temp(i1, i2, cnt_samp, sim_i1, sim_i2), priors, ...
                        hess_temp{i1, i2, cnt_samp, sim_i1, sim_i2}, NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                end
            end
        end
    end
end

AIC_temp = nan*ll_temp;
AIC_temp(1,1,:,:,:) = 2*ll_temp(1,1,:,:,:)+2*8;%+(2*8^2+2*8)/(432-8-1);
AIC_temp(1,2:end,:,:,:) = 2*ll_temp(1,2:end,:,:,:)+2*10;%+(2*10^2+2*10)/(432-10-1);
AIC_temp(2:end,1,:,:,:) = 2*ll_temp(2:end,1,:,:,:)+2*10;%+(2*10^2+2*10)/(432-10-1);
AIC_temp(2:end,2:end,:,:,:) = 2*ll_temp(2:end,2:end,:,:,:)+2*12;%+(2*12^2+2*12)/(432-12-1);

BIC_temp = nan*ll_temp;
BIC_temp(1,1,:,:,:) = 2*ll_temp(1,1,:,:,:)+log(432)*8;
BIC_temp(1,2:end,:,:,:) = 2*ll_temp(1,2:end,:,:,:)+log(432)*10;
BIC_temp(2:end,1,:,:,:) = 2*ll_temp(2:end,1,:,:,:)+log(432)*10;
BIC_temp(2:end,2:end,:,:,:) = 2*ll_temp(2:end,2:end,:,:,:)+log(432)*12;

conf_mat = zeros(16);
for s=1:nSample
    for m=1:4
        for n=1:4
            [M, I] = min(permute(AIC_temp(:,:,s,m,n),[2 1 3 4 5]), [],[1 2],'linear');
            conf_mat((m-1)*4+n,I) = conf_mat((m-1)*4+n,I)+1/nSample;
        end
    end
end

figure;
imagesc(conf_mat);
xticks(1:16)
xticklabels(xlabels)
% xticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
xlabel('Fit Model')
yticks(1:16)
yticklabels(xlabels)
% yticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
ylabel('Simulated Model')
colorbar()
title('Confusion Matrix')
[txs, tys] = meshgrid(1:16, 1:16);
text(txs(:)-0.2, tys(:), string(conf_mat(:))')

%%
% for s=1:nSample
%     for m=1:2
%         for n=1:2
%             [M, I] = min(permute(ll_temp(m,n,s,:,:),[1 2 3 5 4]), [],[4 5],'linear');
%             inv_mat(I,(m-1)*2+n) = inv_mat(I,(m-1)*2+n)+1/nSample;
%         end
%     end
% end

inv_mat = conf_mat./sum(conf_mat, 1);

figure;
imagesc(inv_mat);
xticks(1:16)
% xticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
xticklabels(xlabels)
xlabel('Fit Model')
yticks(1:16)
% yticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
yticklabels(xlabels)
ylabel('Simulated Model')
colorbar()
title('Inversion Matrix')

[txs, tys] = meshgrid(1:16, 1:16);
text(txs(:), tys(:), string(round(inv_mat(:), 2)'))

% calculate difference between null and given distribution