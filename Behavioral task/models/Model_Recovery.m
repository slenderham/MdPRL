clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs/")
addpath("../PRLexp/SubjectData/")
addpath("../utils")
addpath("../../PRLexpv3_5v2/")

%%

% subjects = {...
%     'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', ...
%     'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', ...
%     'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', ...
%     'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', ...
%     'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', ...
%     'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'CC', 'DD', ...
%     'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', ...
%     'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', ...
%     'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ'} ;

% subjects = {'AA'};

attn_modes = {'const', 'diff', 'sum', 'max'};
len_i_1 = length(attn_modes);
len_i_2 = length(attn_modes);

all_model_names = ["fMLchoiceLL_RL2ftdecayattn", ...
    "fMLchoiceLL_RL2ftobjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn_constrained"];

nSamples = 100;
trialLength = 432;
nrep = 15;

op = optimset('Display', 'none');
poolobj = parpool('local', 16);

%%
% disp('Making New Stimuli')
% for cnt_samp = 1:nSamples
%     disp(['Sample: ', num2str(cnt_samp)])
%     fGenerateInputIndividual(num2str(cnt_samp,'%03.f'), 1);
% end

%%
disp('Started Simulation')
for cnt_samp = 1:nSamples
    disp(['Sample: ', num2str(cnt_samp)])
    inputname   = ['../PRLexp/inputs_sim/input_', num2str(cnt_samp,'%03.f') , '.mat'] ;

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

    for i1 = 1:len_i_1
        for i2 = 1:len_i_2
            sesdata = struct();
            sesdata.sig     = 1 ;
            sesdata.input   = input ;
            sesdata.expr    = expr ;
            sesdata.NtrialsShort = expr.NtrialsShort ;
            sesdata.flagUnr = 1 ;

            % RL2 Feature decay
            sesdata.flag_couple = 0 ;
            sesdata.flag_updatesim = 0 ;
            sesdata.flagSepAttn = 1;

            sesdata.NparamBasic = 3 ;
            if sesdata.flagUnr==1
                sesdata.Nalpha = 2 ;
            else
                sesdata.Nalpha = 1 ;
            end
            sesdata.Nbeta = 2;
            sesdata.attn_mode_choice = attn_modes{i1};
            sesdata.attn_mode_learn = attn_modes{i2};
            ipars = [2*rand(1)-1 ...
                exp(log(1)+rand(1)*(log(20)-log(1))) ...
                unifrnd(0.005, 0.04, [1, 1]) ...
                unifrnd(0.05, 0.4, [1, 2]) ...
                exp(log(1)+rand(1, 2)*(log(20)-log(1)))];

            [Cs, Rs, Vs, As] = fMLchoiceSim_RL2ftdecayattn(ipars, sesdata);
            all_ft_Cs(cnt_samp, i1, i2, 1, 1:trialLength) = Cs;
            all_ft_Rs(cnt_samp, i1, i2, 1, 1:trialLength) = Rs;

            % RL2 Feature+Obj decay
            sesdata.NparamBasic = 4 ;

            if sesdata.flagUnr==1
                sesdata.Nalpha = 4 ;
            else
                sesdata.Nalpha = 2 ;
            end
            sesdata.Nbeta = 2;
            sesdata.attn_mode_choice = attn_modes{i1};
            sesdata.attn_mode_learn = attn_modes{i2};

            ipars = [2*rand(1)-1 ...
                exp(log(1)+rand(1, 2)*(log(20)-log(1))) ...
                unifrnd(0.005, 0.04, [1, 1]) ...
                unifrnd(0.05, 0.4, [1, 4]) ...
                exp(log(1)+rand(1, 2)*(log(20)-log(1)))];

            [Cs, Rs, Vs, As] = fMLchoiceSim_RL2ftobjdecayattn(ipars, sesdata);
            all_ftobj_Cs(cnt_samp, i1, i2, 1, 1:trialLength) = Cs;
            all_ftobj_Rs(cnt_samp, i1, i2, 1, 1:trialLength) = Rs;

            % RL2 conjunction decay
            sesdata.NparamBasic = 4 ;

            if sesdata.flagUnr==1
                sesdata.Nalpha = 4 ;
            else
                sesdata.Nalpha = 2 ;
            end
            sesdata.Nbeta = 4;
            sesdata.attn_mode_choice = attn_modes{i1};
            sesdata.attn_mode_learn = attn_modes{i2};

            ipars = [2*rand(1)-1 ...
                exp(log(1)+rand(1, 2)*(log(20)-log(1)))...
                unifrnd(0.005, 0.04, [1, 1]) ...
                unifrnd(0.05, 0.4, [1, 4]) ...
                exp(log(1)+rand(1, 4)*(log(20)-log(1)))];

            [Cs, Rs, Vs, As] = fMLchoiceSim_RL2conjdecayattn(ipars, sesdata);
            all_ftconj_Cs(cnt_samp, i1, i2, 1, 1:trialLength) = Cs;
            all_ftconj_Rs(cnt_samp, i1, i2, 1, 1:trialLength) = Rs;


            % RL2 conjunction decay constrained attn
            sesdata.NparamBasic = 4 ;

            if sesdata.flagUnr==1
                sesdata.Nalpha = 4 ;
            else
                sesdata.Nalpha = 2 ;
            end
            sesdata.Nbeta = 4;
            sesdata.attn_mode_choice = attn_modes{i1};
            sesdata.attn_mode_learn = attn_modes{i2};

            ipars = [2*rand(1)-1 ...
                exp(log(1)+rand(1, 2)*(log(20)-log(1)))...
                unifrnd(0.005, 0.04, [1, 1]) ...
                unifrnd(0.05, 0.4, [1, 4]) ...
                exp(log(1)+rand(1, 4)*(log(20)-log(1)))];

            [Cs, Rs, Vs, As] = fMLchoiceSim_RL2conjdecayattn_constrained(ipars, sesdata);
            all_ftconj_constr_Cs(cnt_samp, i1, i2, 1, 1:trialLength) = Cs;
            all_ftconj_constr_Rs(cnt_samp, i1, i2, 1, 1:trialLength) = Rs;

        end
    end
end


all_Cs = cat(4, all_ft_Cs, all_ftobj_Cs, all_ftconj_Cs, all_ftconj_constr_Cs);
all_Rs = cat(4, all_ft_Rs, all_ftobj_Rs, all_ftconj_Rs, all_ftconj_constr_Rs);

%%

% disp('Calculating Performance')
% 
% % perfMeans = zeros(len_i_1, len_i_2, 4);
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
%             perfMeans(i1, i2, :) = perfMeans(i1, i2, :) + reshape(nanmean(choice_better(:,ntrialPerf), 2)/nSamples, [1, 1, 4]) ;
%         end
%     end
% end
% 
% disp(perfMeans)


%% Plot Performance

% perfMean = nanmean(choice_better(:,:,:,:,ntrialPerf), [1, 5]);
% perfStd = nanstd(choice_better(:,:,:,:,ntrialPerf), [], [1, 5]);
% 
% b = bar(reshape(perfMean, 16, 4));hold on;
% nbars = 4;ngroups=16;
% x = nan(nbars, ngroups);
% for i = 1:nbars
%     x(i,:) = b(i).XEndPoints;
% end
% er = errorbar(x', reshape(perfMean, 16, 4), reshape(perfStd, 16, 4)/sqrt(length(ntrialPerf*nSamples)));
% for i=1:4
%     er(i).Color = [0 0 0];
%     er(i).LineStyle = 'none';
% end
% % legend(er, {'','','',''})
% legend(b, {'F', 'F+O', 'F+C_{untied}', 'F+C_{tied}'}')
% 
% xticks(1:16)
% xticklabels(xlabels)
% 
% ylim([0.4, 0.8])
% ylabel('Performance')

%% model fitting
disp('Simulation complete, now fitting')

parfor cnt_samp = 1:nSamples
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

    % iterate over the models used to simulate
    for sim_i1 = 1:4
        for sim_i2 = 1:4
            for sim_dim = 4

                %                 fvalminRL2_ft_attn = ones(len_i_1, len_i_2)*10000;
                %                 fvalminRL2_ftobj_attn = ones(len_i_1, len_i_2)*10000;
                %                 fvalminRL2_ftconj_attn = ones(len_i_1, len_i_2)*10000;
                fvalminRL2_ftconj_attn_constr = ones(len_i_1, len_i_2)*10000;

                % iterate over the models to fit
                for cnt_rep  = 1:nrep
                    disp(strcat('Sample: ', num2str(cnt_samp), ...
                        ', Repeat: ', num2str(cnt_rep), ...
                        ', Attn Choice: ', attn_modes{sim_i1}, ...
                        ', Attn Learn: ', attn_modes{sim_i2}, ...
                        ', Dim: ', all_model_names(sim_dim)));
                    for i1 = 1:4
                        for i2 = 1:4
                            sesdata = struct();
                            sesdata.sig     = 1 ;
                            sesdata.input   = input ;
                            sesdata.expr    = expr ;
                            sesdata.NtrialsShort = expr.NtrialsShort ;
                            sesdata.flagUnr = 1 ;
                            sesdata.flag_couple = 0 ;
                            sesdata.flag_updatesim = 0 ;
                            sesdata.flagSepAttn = 1;

                            % RL2 Feature decay
                            %
                            %                             NparamBasic = 3 ;
                            %                             if sesdata.flagUnr==1
                            %                                 sesdata.Nalpha = 2 ;
                            %                             else
                            %                                 sesdata.Nalpha = 1 ;
                            %                             end
                            %                             sesdata.Nbeta = 2;
                            %                             sesdata.attn_mode_choice = attn_modes{i1};
                            %                             sesdata.attn_mode_learn = attn_modes{i2};
                            %                             sesdata.results.choice = all_Cs{cnt_samp, sim_i1, sim_i2, sim_dim};
                            %                             sesdata.results.reward = all_Rs{cnt_samp, sim_i1, sim_i2, sim_dim};
                            %
                            %                             ipar= rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                            %                             ll = @(x)sum(fMLchoiceLL_RL2ftdecayattn(x, sesdata));
                            %                             lbs = [-50,  0, 0, 0, 0,  0,  0];
                            %                             ubs = [ 50, 50, 1, 1, 1, 50, 50];
                            %                             [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                            %                             if fval <= fvalminRL2_ft_attn(i1, i2)
                            %                                 fvalminRL2_ft_attn(i1, i2) = fval ;
                            %                                 mlparRL2ft_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                            %                                 mlparRL2ft_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(100) = fval ;
                            %                                 mlparRL2ft_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(101) = fval./length(sesdata.results.reward) ;
                            %                                 mlparRL2ft_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(102) = output.iterations;
                            %                                 mlparRL2ft_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(103) = exitflag ;
                            %                             end

                            % RL2 Feature+Obj decay
                            %                             NparamBasic = 4 ;
                            %
                            %                             if sesdata.flagUnr==1
                            %                                 sesdata.Nalpha = 4 ;
                            %                             else
                            %                                 sesdata.Nalpha = 2 ;
                            %                             end
                            %                             sesdata.Nbeta = 2;
                            %                             sesdata.attn_mode_choice = attn_modes{i1};
                            %                             sesdata.attn_mode_learn = attn_modes{i2};
                            %                             sesdata.results.choice = all_Cs{cnt_samp, sim_i1, sim_i2, sim_dim};
                            %                             sesdata.results.reward = all_Rs{cnt_samp, sim_i1, sim_i2, sim_dim};
                            %
                            %                             ipar= rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                            %                             ll = @(x)sum(fMLchoiceLL_RL2ftobjdecayattn(x, sesdata));
                            %                             lbs = [-50,  0,  0, 0, 0, 0, 0, 0,  0,  0];
                            %                             ubs = [ 50, 50, 50, 1, 1, 1, 1, 1, 50, 50];
                            %                             [xpar, fval, exitflag, output] = fmincon(ll, ipar, [],[],[],[], lbs, ubs, [], op) ;
                            %                             if fval <= fvalminRL2_ftobj_attn(i1, i2)
                            %                                 fvalminRL2_ftobj_attn(i1, i2) = fval ;
                            %                                 mlparRL2ftobj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                            %                                 mlparRL2ftobj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(100) = fval ;
                            %                                 mlparRL2ftobj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(101) = fval./length(sesdata.results.reward) ;
                            %                                 mlparRL2ftobj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(102) = output.iterations;
                            %                                 mlparRL2ftobj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(103) = exitflag ;
                            %                             end

                            % RL2 conjunction decay
                            %                             NparamBasic = 4 ;
                            %
                            %                             if sesdata.flagUnr==1
                            %                                 sesdata.Nalpha = 4 ;
                            %                             else
                            %                                 sesdata.Nalpha = 2 ;
                            %                             end
                            %                             sesdata.Nbeta = 4;
                            %                             sesdata.attn_mode_choice = attn_modes{i1};
                            %                             sesdata.attn_mode_learn = attn_modes{i2};
                            %                             sesdata.results.choice = all_Cs{cnt_samp, sim_i1, sim_i2, sim_dim};
                            %                             sesdata.results.reward = all_Rs{cnt_samp, sim_i1, sim_i2, sim_dim};
                            %
                            %                             ipar= rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                            %                             ll = @(x)sum(fMLchoiceLL_RL2conjdecayattn(x, sesdata));
                            %                             lbs = [-50,  0,  0, 0, 0, 0, 0, 0,  0,  0,  0,  0];
                            %                             ubs = [ 50, 50, 50, 1, 1, 1, 1, 1, 50, 50, 50, 50];
                            %                             [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                            %                             if fval <= fvalminRL2_ftconj_attn(i1, i2)
                            %                                 fvalminRL2_ftconj_attn(i1, i2) = fval ;
                            %                                 mlparRL2conj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                            %                                 mlparRL2conj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(100) = fval ;
                            %                                 mlparRL2conj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(101) = fval./length(sesdata.results.reward) ;
                            %                                 mlparRL2conj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(102) = output.iterations;
                            %                                 mlparRL2conj_decay_attn{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(103) = exitflag ;
                            %                             end

                            % RL2 conjunction decay constrained attn
                            NparamBasic = 4 ;

                            if sesdata.flagUnr==1
                                sesdata.Nalpha = 4 ;
                            else
                                sesdata.Nalpha = 2 ;
                            end
                            sesdata.Nbeta = 4;
                            sesdata.attn_mode_choice = attn_modes{i1};
                            sesdata.attn_mode_learn = attn_modes{i2};
                            sesdata.results.choice = all_Cs(cnt_samp, sim_i1, sim_i2, sim_dim, :);
                            sesdata.results.reward = all_Rs(cnt_samp, sim_i1, sim_i2, sim_dim, :);

                            ipar= rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                            ll = @(x)sum(fMLchoiceLL_RL2conjdecayattn_constrained(x, sesdata));
                            lbs = [-50,  0,  0, 0, 0, 0, 0, 0,  0,  0,  0,  0];
                            ubs = [ 50, 50, 50, 1, 1, 1, 1, 1, 50, 50, 50, 50];
                            [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                            if fval <= fvalminRL2_ftconj_attn_constr(i1, i2)
                                fvalminRL2_ftconj_attn_constr(i1, i2) = fval ;
                                mlparRL2conj_decay_attn_constr{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                                mlparRL2conj_decay_attn_constr{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(100) = fval ;
                                mlparRL2conj_decay_attn_constr{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(101) = fval./length(sesdata.results.reward) ;
                                mlparRL2conj_decay_attn_constr{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(102) = output.iterations;
                                mlparRL2conj_decay_attn_constr{i1, i2, cnt_samp, sim_i1, sim_i2, sim_dim}(103) = exitflag ;
                            end
                        end
                    end
                end
            end
        end
    end
end

cd ../files
save RPL2Analysis_Attention_model_recovery
cd ../models