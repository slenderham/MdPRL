clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_sim/")
addpath("../utils")
addpath("../utils/bads")
addpath("../utils/DERIVESTsuite/DERIVESTsuite/")
addpath("../../PRLexpv3_5v2/")

%%

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];

all_model_names_legend = ["F", "F+O", "F+C_{untied}", "F+C_{feat attn}", "F+C_{tied}"];
attn_modes_legend = strcat(attn_modes(:,1),"X",attn_modes(:,2));
[all_model_names_legend, attn_modes_legend] = meshgrid(all_model_names_legend, attn_modes_legend);
all_legends = strcat(all_model_names_legend(:), "X", attn_modes_legend(:));

all_model_names = ["fMLchoiceLL_RL2ftdecayattn", ...
    "fMLchoiceLL_RL2ftobjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn_onlyfattn", ...
    "fMLchoiceLL_RL2conjdecayattn_constrained"];

all_model_Nparambasic = [3, 4, 4, 4, 4];
all_model_Nalphas = [2, 2, 2, 2, 2];
all_model_Nbetas = [1, 1, 1, 1, 1];

bound_eps = 0;
bias_bound = 5;
p_bias_bound = 5;
temp_bound = 100;
p_temp_bound = 100;
attn_temp_bound = 100;
p_attn_temp_bound = 100;

all_lbs = {...
    [-bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps]};

all_ubs = {...
    [bias_bound, temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound]};

all_plbs = {...
    [-p_bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-p_bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-p_bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-p_bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps], ...
    [-p_bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps]};

all_pubs = {...
    [p_bias_bound, p_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound]};

all_sim_par_samples = {...
    @() [randn(1)/10 exp(log(1)+rand(1)*(log(100)-log(1))) ...
         unifrnd(0.005, 0.1, [1, 1]) unifrnd(0.05, 0.95, [1, 2])...
         exp(log(1)+rand(1)*(log(100)-log(1)))], ...
    @() [randn(1)/10 exp(log(1)+rand(1)*(log(100)-log(1))) rand(1) ...
         unifrnd(0.005, 0.1, [1, 1]) unifrnd(0.05, 0.95, [1, 2]) ...
         exp(log(1)+rand(1)*(log(100)-log(1)))], ...
    @() [randn(1)/10 exp(log(1)+rand(1)*(log(100)-log(1))) rand(1) ...
         unifrnd(0.005, 0.1, [1, 1]) unifrnd(0.05, 0.95, [1, 2]) ...
         exp(log(1)+rand(1)*(log(100)-log(1)))], ...
    @() [randn(1)/10 exp(log(1)+rand(1)*(log(100)-log(1))) rand(1) ...
         unifrnd(0.005, 0.1, [1, 1]) unifrnd(0.05, 0.95, [1, 2]) ...
         exp(log(1)+rand(1)*(log(100)-log(1)))], ...
    @() [randn(1)/10 exp(log(1)+rand(1)*(log(100)-log(1))) rand(1) ...
         unifrnd(0.005, 0.1, [1, 1]) unifrnd(0.05, 0.95, [1, 2]) ...
         exp(log(1)+rand(1)*(log(100)-log(1)))], ...
    };

nSamples = 100;
trialLength = 432;
nrep = 50;

op = optimset('Display', 'off');

%%
% disp('Making New Stimuli')
% for cnt_samp = 1:nSamples
%     disp(['Sample: ', num2str(cnt_samp)])
%     fGenerateInputIndividual(num2str(cnt_samp,'%03.f'), 1);
% end

%%

poolobj = parpool('local', 16);

for m = 1:length(all_model_names)
    disp("=======================================================");
    disp(strcat("Simulating model ", all_model_names(m)));
    for a = 1:length(attn_modes)
        disp("-------------------------------------------------------");
        disp(strcat("Simulating attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        parfor cnt_samp = 1:nSamples
%             disp(strcat("Simulating subject ", num2str(cnt_samp)));
            inputname   = ['../PRLexp/inputs_sim/input_', num2str(cnt_samp,'%03.f') , '.mat'] ;

            inputs_struct = load(inputname);
            input = inputs_struct.input;
            
            sesdata = struct();
            sesdata.input   = input ;
            sesdata.results.choice = nan*zeros(trialLength);
            sesdata.results.reward = nan*zeros(trialLength);

            sesdata.expr.shapeMap = repmat([1 2 3 ;
                1 2 3 ;
                1 2 3 ], 1,1,3) ;

            sesdata.expr.colorMap = repmat([1 1 1 ;
                2 2 2 ;
                3 3 3], 1,1,3) ;

            sesdata.expr.patternMap(:,:,1) = ones(3,3) ;
            sesdata.expr.patternMap(:,:,2) = 2*ones(3,3) ;
            sesdata.expr.patternMap(:,:,3) = 3*ones(3,3) ;

            sesdata.NtrialsShort = trialLength;
            sesdata.flagUnr = 1;

            sesdata.flag_couple = 0 ;
            sesdata.flag_updatesim = 0 ;

            % load parameter numbers
            NparamBasic = all_model_Nparambasic(m);
            sesdata.Nalpha = all_model_Nalphas(m);

            if a==1
                sesdata.Nbeta = 0;
            else
                sesdata.Nbeta = all_model_Nbetas(m);
            end

            % load attn type (const, diff, sum, max) and attn
            % time(none, choice, learning, both)
            sesdata.attn_op = attn_modes(a,1);
            sesdata.attn_time = attn_modes(a,2);

            sim_ll = 100000
            while sim_ll > -trialLength*log(0.53)
                % load lb and ub, initialize
                ipars = all_sim_par_samples{m}();
                ipars = ipars(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
    
                % load model likelihood func and optimize
                ll = str2func(all_model_names(m));
                [fval, latents] = ll(ipars, sesdata);
                sim_ll = sum(fval);
    
                sim_results{m, a, cnt_samp}.params = ipars;
                sim_results{m, a, cnt_samp}.fval = sum(fval);
                sim_results{m, a, cnt_samp}.C = latents.C;
                sim_results{m, a, cnt_samp}.R = latents.R;
            end
        end
    end
end

%%

% disp('Calculating Performance')
% 
% % perfMeans = zeros(len_i_1, len_i_2, 4);
% 
% ntrialPerf = 33:432;
% 
% for cnt_samp = 1:nSamples
%     inputname   = ['../PRLexp/inputs_sim/input_', num2str(cnt_samp,'%03.f') , '.mat'] ;
%     input_struct =  load(inputname);
%     [~, idxMax] = max(input_struct.expr.prob{1}(input_struct.input.inputTarget)) ;
% %     disp(['Subject ' num2str(cnt_samp)])
%     for m = 1:length(all_model_names)
%         for a = 1:length(attn_modes)
%             choice_better(cnt_samp, m, a, :) = idxMax==squeeze(sim_results{m, a, cnt_samp}.C);
%         end
%     end
% end
% 
% disp(squeeze(mean(choice_better, [1 4])));


%% Plot Performance

% perfMean = squeeze(mean(choice_better(:,:,:,ntrialPerf), [1, 4]))';
% perfStd = squeeze(std(choice_better(:,:,:,ntrialPerf), [], [1, 4]))';
% 
% b = bar(perfMean);hold on;
% nbars = 5;ngroups=10;
% x = nan(nbars, ngroups);
% 
% for i = 1:nbars
%     x(i,:) = b(i).XEndPoints;
% end
% 
% er = errorbar(x', perfMean, perfStd/sqrt(length(ntrialPerf*nSamples)));
% 
% for i=1:5
%     er(i).Color = [0 0 0];
%     er(i).LineStyle = 'none';
% end
% % legend(er, {'','','',''})
% legend(b, {'F', 'F+O', 'F+C_{untied}', 'F+C_{fa}', 'F+C_{tied}'}')
% 
% xticks(1:16)
% xticklabels(attn_modes_legend(:,1))
% 
% ylim([0.4, 0.8])
% ylabel('Performance')

%% model fitting
disp('Simulation complete, now fitting')

for m = 5
    disp("=======================================================");
    disp(strcat("Fitting model ", all_model_names(m)));
    for a = 3
        disp("-------------------------------------------------------");
        disp(strcat("Fitting attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        for sim_m = 5
            for sim_a = 3
                tic
                parfor cnt_samp = 1:nSamples
                    disp(strcat('Sample: ', num2str(cnt_samp), ...
                                ', Model: ', all_model_names(sim_m), ...
                                ', Attn: ', attn_modes(sim_a, 1), " X ", attn_modes(sim_a, 2)));

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
                    
%                     disp(sim_results{sim_m, sim_a, cnt_samp}.fval)
%                     disp(sim_results{sim_m, sim_a, cnt_samp}.params)
                    minfval = 1000000;
                    for cnt_rep = 1:nrep

                        sesdata = struct();
                        sesdata.sig     = 1 ;
                        sesdata.input   = input ;
                        sesdata.expr    = expr ;
                        sesdata.NtrialsShort = expr.NtrialsShort ;
                        sesdata.flagUnr = 1;
                        sesdata.flag_couple = 0 ;
                        sesdata.flag_updatesim = 0 ;

                        sesdata.results.choice = sim_results{sim_m, sim_a, cnt_samp}.C;
                        sesdata.results.reward = sim_results{sim_m, sim_a, cnt_samp}.R;

                        % load parameter numbers
                        NparamBasic = all_model_Nparambasic(m);
                        sesdata.Nalpha = all_model_Nalphas(m);

                        if a==1
                            sesdata.Nbeta = 0;
                        else
                            sesdata.Nbeta = all_model_Nbetas(m);
                        end

                        % load attn type (const, diff, sum, max) and attn
                        % time(none, choice, learning, both)
                        sesdata.attn_op = attn_modes(a,1);
                        sesdata.attn_time = attn_modes(a,2);

                        % load lb and ub, initialize
                        plbs = all_plbs{m};
                        pubs = all_pubs{m};
                        plbs = plbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                        pubs = pubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                        ipar = plbs+rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta).*(pubs-plbs);
                        
                        lbs = all_lbs{m};
                        ubs = all_ubs{m};
                        lbs = lbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                        ubs = ubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
%                         lbs(1:end-1) = ipar(1:end-1); plbs(1:end-1) = ipar(1:end-1);
%                         ubs(1:end-1) = ipar(1:end-1); pubs(1:end-1) = ipar(1:end-1);
                        % load model likelihood func and optimize
                        ll = str2func(all_model_names(m));
                        ll = @(x)sum(ll(x, sesdata));

                        [xpar, fval, exitflag, output] = bads(ll, ipar, lbs, ubs, plbs, pubs, [], op) ;

                        if fval <= minfval
                            minfval = fval ;
                            fit_results{m, a, cnt_samp, sim_m, sim_a}.params = xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                            fit_results{m, a, cnt_samp, sim_m, sim_a}.fval = fval ;
                            fit_results{m, a, cnt_samp, sim_m, sim_a}.iters = output.iterations;
                            fit_results{m, a, cnt_samp, sim_m, sim_a}.exitflag = exitflag ;
                        end
                    end
                end
                toc
            end
        end
    end
end


cd ../files
save RPL2Analysis_Attention_model_recovery
cd ../models
