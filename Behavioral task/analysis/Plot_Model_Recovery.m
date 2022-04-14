clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs/")
addpath("../PRLexp/SubjectData/")
addpath("../utils")
addpath("../files/")
addpath("../../PRLexpv3_5v2/")

load('RPL2Analysis_Attention_model_recovery.mat')

nSample = 100;

ll_temp=0;
for i1=1:2
    for i2=1:2
        for sim_i1=1:2
            for sim_i2=1:2
                for cnt_samp=1:nSample
                    ll_temp(i1, i2, cnt_samp, sim_i1, sim_i2) = mlparRL2conj_decay_attn_constr{i1+1, i2+1, cnt_samp, sim_i1+1, sim_i2+1, 4}(100);
                end
            end
        end
    end
end

conf_mat = zeros(4);
for s=1:nSample
    for m=1:2
        for n=1:2
            [M, I] = min(permute(ll_temp(:,:,s,m,n),[2 1 3 4 5]), [],[1 2],'linear');
            conf_mat((m-1)*2+n,I) = conf_mat((m-1)*2+n,I)+1/nSample;
        end
    end
end

figure;
imagesc(conf_mat);
xticks(1:4)
xticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
xlabel('Fit Model')
yticks(1:4)
yticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
ylabel('Simulated Model')
colorbar()
title('Confusion Matrix')
[txs, tys] = meshgrid(1:4, 1:4);
text(txs(:), tys(:), string(conf_mat(:))')

%% 
inv_mat = zeros(4);
for s=1:nSample
    for m=1:2
        for n=1:2
            [M, I] = min(permute(ll_temp(m,n,s,:,:),[1 2 3 5 4]), [],[4 5],'linear');
            inv_mat(I,(m-1)*2+n) = inv_mat(I,(m-1)*2+n)+1/nSample;
        end
    end
end

figure;
imagesc(inv_mat);
xticks(1:4)
xticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
xlabel('Fit Model')
yticks(1:4)
yticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
ylabel('Simulated Model')
colorbar()
title('Inversion Matrix')

[txs, tys] = meshgrid(1:4, 1:4);
text(txs(:), tys(:), string(inv_mat(:))')