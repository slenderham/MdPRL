function [loglikehood, latents] = fMLchoiceLL_RL2ftdecayattn(xpar, sesdata)
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
mag = xpar(2) ;
decay = xpar(3) ;

alpha_rewColor      = xpar(NparamBasic+1) ;
alpha_rewShape      = xpar(NparamBasic+1) ;
alpha_rewPattern    = xpar(NparamBasic+1) ;
NparamWithLR = NparamBasic+1;
if sesdata.flagUnr==1
    alpha_unrColor      = xpar(NparamBasic+2) ;
    alpha_unrShape      = xpar(NparamBasic+2) ;
    alpha_unrPattern    = xpar(NparamBasic+2) ;
    NparamWithLR = NparamBasic+2;
else
    alpha_unrColor      = alpha_rewColor ;
    alpha_unrShape      = alpha_rewShape ;
    alpha_unrPattern    = alpha_rewPattern ;
end

if strcmp(sesdata.attn_time, "none")
    beta_attn = 1;
else
    beta_attn = xpar(NparamWithLR+1);
end

shapeMap        = sesdata.expr.shapeMap ;
colorMap        = sesdata.expr.colorMap ;
patternMap      = sesdata.expr.patternMap ;
inputTarget     = sesdata.input.inputTarget ;
correcttrials   = sesdata.results.reward ;
choicetrials    = sesdata.results.choice ;
flag_couple     = sesdata.flag_couple ;
flag_updatesim  = sesdata.flag_updatesim ;
ntrials         = length(choicetrials) ;
inputRewards    = sesdata.input.inputReward ;

v = (0.5*ones(9,1)) ;
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

    attn_w = attention_weights(v, ...
        [idx_shape(1), idx_color(1), idx_pattern(1)], ...
        [idx_shape(2), idx_color(2), idx_pattern(2)], ...
        sesdata.attn_op, beta_attn);

    if strcmp(sesdata.attn_time, "C")
        attn_w_choice = attn_w;
        attn_w_learn = ones(1, 3)/3;
    elseif strcmp(sesdata.attn_time, "L")
        attn_w_choice = ones(1, 3)/3;
        attn_w_learn = attn_w;
    elseif strcmp(sesdata.attn_time, "CL")
        attn_w_choice = attn_w;
        attn_w_learn = attn_w;
    elseif strcmp(sesdata.attn_time, "none")
        attn_w_choice = ones(1, 3)/3;
        attn_w_learn = ones(1, 3)/3;
    end

    vsum(1) = attn_w_choice(1)*v(idx_shape(1)) ...
            + attn_w_choice(2)*v(idx_color(1)) ...
            + attn_w_choice(3)*v(idx_pattern(1));
    vsum(2) = attn_w_choice(1)*v(idx_shape(2)) ...
            + attn_w_choice(2)*v(idx_color(2)) ...
            + attn_w_choice(3)*v(idx_pattern(2));

    logit = mag*(vsum(2)-vsum(1))-BiasL;

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
        else
            loglikehood(cnt_trial) =  - logsigmoid(-logit) ;
        end
        if sesdata.use_rpe
            rpe = abs(correct - vsum(choice));
        else
            rpe = 1;
        end
    end

    if correct
        idxC = idx_color(choice) ;
        idxW = idx_color(3-choice) ;
        v = decayV(v, 3+find(3+[1:3]~=idx_color(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0, flag_updatesim) ;
        v = update(v, idxC, idxW, alpha_rewColor*attn_w_learn(2)*rpe) ;

        idxC = idx_shape(choice) ;
        idxW = idx_shape(3-choice) ;
        v = decayV(v, find([1:3]~=idx_shape(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0, flag_updatesim) ;
        v = update(v, idxC, idxW, alpha_rewShape*attn_w_learn(1)*rpe) ;

        idxC = idx_pattern(choice) ;
        idxW = idx_pattern(3-choice) ;
        v = decayV(v, 6+find(6+[1:3]~=idx_pattern(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0, flag_updatesim) ;
        v = update(v, idxC, idxW, alpha_rewPattern*attn_w_learn(3)*rpe) ;
    else
        idxW = idx_color(choice) ;
        idxC = idx_color(3-choice) ;
        v = decayV(v, 3+find([4:6]~=idx_color(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0, flag_updatesim) ;
        v = update(v, idxC, idxW, alpha_unrColor*attn_w_learn(2)*rpe) ;

        idxW = idx_shape(choice) ;
        idxC = idx_shape(3-choice) ;
        v = decayV(v, find([1:3]~=idx_shape(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0, flag_updatesim) ;
        v = update(v, idxC, idxW, alpha_unrShape*attn_w_learn(1)*rpe) ;

        idxW = idx_pattern(choice) ;
        idxC = idx_pattern(3-choice) ;
        v = decayV(v, 6+find([7:9]~=idx_pattern(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0, flag_updatesim) ;
        v = update(v, idxC, idxW, alpha_unrPattern*attn_w_learn(3)*rpe) ;
    end
    if flag_couple
        if correctunCh
            idxC = idx_color(choiceunCh) ;
            idxW = idx_color(3-choiceunCh) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            v = update(v, idxC, idxW, alpha_rewColor*attn_w_learn(2)*rpe) ;

            idxC = idx_shape(choiceunCh) ;
            idxW = idx_shape(3-choiceunCh) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            v = update(v, idxC, idxW, alpha_rewShape*attn_w_learn(1)*rpe) ;

            idxC = idx_pattern(choiceunCh) ;
            idxW = idx_pattern(3-choiceunCh) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            v = update(v, idxC, idxW, alpha_rewPattern*attn_w_learn(3)*rpe) ;
        else
            idxW = idx_color(choiceunCh) ;
            idxC = idx_color(3-choiceunCh) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            v = update(v, idxC, idxW, alpha_unrColor*attn_w_learn(2)*rpe) ;

            idxW = idx_shape(choiceunCh) ;
            idxC = idx_shape(3-choiceunCh) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            v = update(v, idxC, idxW, alpha_unrShape*attn_w_learn(1)*rpe) ;

            idxW = idx_pattern(choiceunCh) ;
            idxC = idx_pattern(3-choiceunCh) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            v = update(v, idxC, idxW, alpha_unrPattern*attn_w_learn(3)*rpe) ;
        end
    end
    latents.V(:,cnt_trial) = v ;
    latents.A(1,1:3,cnt_trial)=attn_w_choice;
    latents.A(2,1:3,cnt_trial)=attn_w_learn;
end
end

% function v = decayV(v, unCh, decay)
%     v(unCh) = v(unCh)*(1-decay) ;
% end
function v = decayV(v, unCh, decay)
v(unCh) = v(unCh) - (v(unCh)-0.5)*(decay) ;
end

function v = update(v, idxC, idxW, Q)
if isempty(idxW) && ~isempty(idxC)
    v(idxC) = v(idxC) + (1-v(idxC)).*Q ;
elseif isempty(idxC) && ~isempty(idxW)
    v(idxW) = v(idxW) - (v(idxW).*Q) ;
elseif ~isempty(idxW) && ~isempty(idxC)
    v(idxC) = v(idxC) + (1-v(idxC)).*Q ;
    v(idxW) = v(idxW) - (v(idxW).*Q) ;
elseif isempty(idxW) && isempty(idxC)
end
end

function [idxW, idxC] = idxcouple(idxW, idxC, rl2_correct, flag_couple, flag_updatesim)
if rl2_correct
    if flag_couple==0
        idxW = [] ;
    elseif flag_couple==1
        if idxW==idxC                                                  % to avoid potentiating and depressing similar V in coupled cases
            idxW= [] ;
            if ~flag_updatesim
                idxC = [] ;
            end
        end
    end
else
    if flag_couple==0
        idxC = [] ;
    elseif flag_couple==1
        if idxW==idxC                                                  % to avoid potentiating and depressing similar V in coupled cases
            idxC= [] ;
            if ~flag_updatesim
                idxW = [] ;
            end
        end
    end
end
end

function [attn] = attention_weights(v, idxes1, idxes2, mode, beta)
    if strcmp(mode, 'diff')
        attn = softmax(beta*abs(v(idxes1)-v(idxes2)));
    elseif strcmp(mode, 'sum')
        attn = softmax(beta*(v(idxes1)+v(idxes2))/2);
    elseif strcmp(mode, 'max')
        attn = softmax(beta*max(v(idxes1), v(idxes2)));
    elseif strcmp(mode, 'const')
        attn = ones(size(idxes1))./size(idxes1, 2);
    else
        error('attn mode not recognized');
    end
end