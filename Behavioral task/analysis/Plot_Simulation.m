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
