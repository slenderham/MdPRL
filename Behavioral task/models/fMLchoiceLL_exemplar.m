function [loglikehood, latents] = fMLchoiceLL_exemplar_static(xpar, sesdata)
%
% DESCRIPTION: fits data to RL(2) model using ML method
%
% INPUT:
% sesdata structure which includes input, experiment and behavioral data
%
% OUTPUT:
% fitted parametres

loglikehood = 0 ;
NparamBasic = 3 ;

BiasL = xpar(1) ;
mag = xpar(2);
decay = xpar(3) ;

alpha_rew     = xpar(NparamBasic+1) ;
NparamWithLR = NparamBasic+1;
if sesdata.flagUnr==1
    alpha_unr      = xpar(NparamBasic+2) ;
    NparamWithLR = NparamBasic+2;
else
    alpha_unr      = alpha_rew;
end

attn_gamma = xpar(NparamWithLR+1) ;

shapeMap        = sesdata.expr.shapeMap ;
colorMap        = sesdata.expr.colorMap ;
patternMap      = sesdata.expr.patternMap ;
inputTarget     = sesdata.input.inputTarget ;
correcttrials   = sesdata.results.reward ;
choicetrials    = sesdata.results.choice ;
ntrials         = length(choicetrials) ;
inputRewards    = sesdata.input.inputReward ;

stim_mem = [];
rwd_mem = []; % rwd X alpha

attn_w = ones(3,1);

for cnt_trial=1:ntrials
    correct = correcttrials(cnt_trial) ;
    choice = choicetrials(cnt_trial) ;
    if ~isnan(choice) && ~isnan(correct)
        correctunCh = inputRewards(3-choice, cnt_trial) ;
        choiceunCh = 3-choice ;
    end

    idx_shape(2)    = shapeMap(inputTarget(2, cnt_trial)) ;
    idx_color(2)    = colorMap(inputTarget(2, cnt_trial))+3 ;
    idx_pattern(2)  = patternMap(inputTarget(2, cnt_trial))+6 ;
    idx_shape(1)    = shapeMap(inputTarget(1, cnt_trial)) ;
    idx_color(1)    = colorMap(inputTarget(1, cnt_trial))+3 ;
    idx_pattern(1)  = patternMap(inputTarget(1, cnt_trial))+6 ;

    if cnt_trial>1
        sims_by_dim = [];
        sims_by_dim(1,:,:) = [(idx_shape(1)~=stim_mem(:,1)) ...
                              (idx_color(1)~=stim_mem(:,2)) ...
                              (idx_pattern(1)~=stim_mem(:,3))];
        sims_by_dim(2,:,:) = [(idx_shape(2)~=stim_mem(:,1)) ...
                              (idx_color(2)~=stim_mem(:,2)) ...
                              (idx_pattern(2)~=stim_mem(:,3))];

        sims_by_dim = reshape(sims_by_dim, [2, 3, cnt_trial-1]);
    
        % sims 2 X dimension X prev trials
    
        sims = sum(reshape(attn_w, [1,3,1]).*sims_by_dim, 2);
        sims = squeeze(exp(-sims));    
        % sims 2 X prev trials
    
        rwd_mem_decay = rwd_mem.*(decay.^(cnt_trial-1:-1:1)); % prev trials

        vsum(1) = sum(sims(1,:).*reshape(rwd_mem_decay, [1,cnt_trial-1]), 2);
        vsum(2) = sum(sims(2,:).*reshape(rwd_mem_decay, [1,cnt_trial-1]), 2);
        
    
        logit = mag*(vsum(2)-vsum(1))-BiasL;
        pure_logit = mag*(vsum(2)-vsum(1));

    else
        logit = 0;
        pure_logit = 0;
    end

    if cnt_trial >= 1
        if isnan(choice) || isnan(correct)  
            pChoiceR = 1./(1+exp(-logit));
        
            choice = binornd(1, pChoiceR)+1;
            choiceunCh = 3-choice;
            
            correct = inputRewards(choice, cnt_trial) ;
            correctunCh = inputRewards(3-choice, cnt_trial) ;
        end
        latents.R(cnt_trial) = correct;
        latents.C(cnt_trial) = choice;
        latents.logits(cnt_trial) = logit;
        if choice == 2
            loglikehood(cnt_trial) =  - logsigmoid(logit) ;
            latents.pure_ll(cnt_trial) = - logsigmoid(pure_logit);
        else
            loglikehood(cnt_trial) =  - logsigmoid(-logit) ;
            latents.pure_ll(cnt_trial) = - logsigmoid(-pure_logit);
        end 
    end

    stim_mem(cnt_trial, :) = [idx_shape(choice), ...
                              idx_color(choice), ...
                              idx_pattern(choice)];
    if correct
        rwd_mem(cnt_trial) = alpha_rew;
    else
        rwd_mem(cnt_trial) = -alpha_unr;
    end

    if cnt_trial>1
        dSL = -reshape(sims(1,:), [1,cnt_trial-1]).*reshape(sims_by_dim(1,:,:), [3, cnt_trial-1]); %3Xtrial
        dSR = -reshape(sims(2,:), [1,cnt_trial-1]).*reshape(sims_by_dim(2,:,:), [3, cnt_trial-1]); %3Xtrial
    
        if choice==2
            dLogit = sum(reshape(rwd_mem_decay, [1,cnt_trial-1]).*(dSR-dSL), 2); %shape=3
            d_attn_w = (attn_gamma.*(2*correct-1).*exp(logsigmoid(-logit))).*dLogit;
        else
            dLogit = sum(reshape(rwd_mem_decay, [1,cnt_trial-1]).*(dSL-dSR), 2); %shape=3
            d_attn_w = (attn_gamma.*(2*correct-1).*exp(logsigmoid(logit))).*dLogit;
        end
        attn_w = attn_w+d_attn_w;
        attn_w = max(attn_w, 0);
    end

    latents.A(cnt_trial,:) = attn_w;
    

end
end
