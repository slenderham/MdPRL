function [loglikehood, latents] = fMLchoiceLL_RL2conjdecay(xpar, sesdata)
%
% DESCRIPTION: fits data to RL(2)obj model using ML method
%
% INPUT: 
% sesdata structure which includes input, experiment and behavioral data
%
% OUTPUT:
% fitted parametres

loglikehood = 0 ;
NparamBasic = 4 ;

BiasL = xpar(1) ;
mag  = xpar(2) ;
omega  = xpar(3) ;
decay = xpar(4) ;

alpha_rewColor      = xpar(NparamBasic+1) ;
alpha_rewShape      = xpar(NparamBasic+1) ;
alpha_rewPattern    = xpar(NparamBasic+1) ; 
alpha_rew           = xpar(NparamBasic+1) ;
if sesdata.flagUnr==1
    alpha_unrColor      = xpar(NparamBasic+2) ;
    alpha_unrShape      = xpar(NparamBasic+2) ;
    alpha_unrPattern    = xpar(NparamBasic+2) ;
    alpha_unr           = xpar(NparamBasic+2) ;
else
    alpha_unrColor      = alpha_rewColor ;
    alpha_unrShape      = alpha_rewShape ;
    alpha_unrPattern    = alpha_rewPattern ;
    alpha_unr           = alpha_rew ;
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
cntD            = sesdata.cntD ;

vf              = (0.5*ones(3,1)) ; 
vc              = (0.5*ones(9,1)) ; 

for cnt_trial=1:ntrials
    latents.V(1:3,cnt_trial) = vf ;
    latents.V(4:12,cnt_trial) = vc ;
    
    correct = correcttrials(cnt_trial) ;
    choice = choicetrials(cnt_trial) ; 
    if ~isnan(choice) && ~isnan(correct)
        correctunCh = inputRewards(3-choice, cnt_trial) ;
        choiceunCh = 3-choice ;
    end
    
    idx_shape(2)    = shapeMap(inputTarget(2, cnt_trial)) ;
    idx_color(2)    = colorMap(inputTarget(2, cnt_trial)) ;
    idx_pattern(2)  = patternMap(inputTarget(2, cnt_trial)) ;
    idx_shape(1)    = shapeMap(inputTarget(1, cnt_trial)) ;
    idx_color(1)    = colorMap(inputTarget(1, cnt_trial)) ;
    idx_pattern(1)  = patternMap(inputTarget(1, cnt_trial)) ;

    if cntD==1
        inputConj(1, cnt_trial) = (idx_pattern(1)-1)*3 + idx_shape(1) ;
        inputConj(2, cnt_trial) = (idx_pattern(2)-1)*3 + idx_shape(2) ;
        vsum(1) = omega*vf(idx_color(1)) + (1-omega)*vc(inputConj(1, cnt_trial));
        vsum(2) = omega*vf(idx_color(2)) + (1-omega)*vc(inputConj(2, cnt_trial));
    elseif cntD==2
        inputConj(1, cnt_trial) = (idx_pattern(1)-1)*3 + idx_color(1) ;
        inputConj(2, cnt_trial) = (idx_pattern(2)-1)*3 + idx_color(2) ;
        vsum(1) = omega*vf(idx_shape(1)) + (1-omega)*vc(inputConj(1, cnt_trial));
        vsum(2) = omega*vf(idx_shape(2)) + (1-omega)*vc(inputConj(2, cnt_trial));
    elseif cntD==3
        inputConj(1, cnt_trial) = (idx_shape(1)-1)*3 + idx_color(1) ;
        inputConj(2, cnt_trial) = (idx_shape(2)-1)*3 + idx_color(2) ;
        vsum(1) = omega*vf(idx_pattern(1)) + (1-omega)*vc(inputConj(1, cnt_trial));
        vsum(2) = omega*vf(idx_pattern(2)) + (1-omega)*vc(inputConj(2, cnt_trial));
    end

    logit = mag*(vsum(2)-vsum(1))-BiasL ;

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
    end
    
    % conjunction
    if correct
        idxC = inputConj(choice, cnt_trial) ;
        idxW = inputConj(3-choice, cnt_trial) ;
        vc = decayV(vc, find([1:9]~=inputConj(choice, cnt_trial)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_rew) ;
    else
        idxC = inputConj(3-choice, cnt_trial) ;
        idxW = inputConj(choice, cnt_trial) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = decayV(vc, find([1:9]~=inputConj(choice, cnt_trial)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_unr) ;
    end
    if flag_couple
        if correctunCh
            idxC = inputConj(choiceunCh, cnt_trial) ;
            idxW = inputConj(3-choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_rew) ;
        else
            idxC = inputConj(3-choiceunCh, cnt_trial) ;
            idxW = inputConj(choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_unr) ;
        end
    end
    
    % feature
    if correct
        if cntD==1
            idxC = idx_color(choice) ;
            idxW = idx_color(3-choice) ;
            vf   = decayV(vf, find([1:3]~=idx_color(choice)), decay) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
            vf           = update(vf, idxC, idxW, alpha_rewColor) ;
        elseif cntD==2
            idxC = idx_shape(choice) ;
            idxW = idx_shape(3-choice) ;
            vf   = decayV(vf, find([1:3]~=idx_shape(choice)), decay) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
            vf           = update(vf, idxC, idxW, alpha_rewShape) ;
        elseif cntD==3
            idxC = idx_pattern(choice) ;
            idxW = idx_pattern(3-choice) ;
            vf   = decayV(vf, find([1:3]~=idx_pattern(choice)), decay) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
            vf           = update(vf, idxC, idxW, alpha_rewPattern) ;
        end
    else
        if cntD==1
            idxW = idx_color(choice) ;
            idxC = idx_color(3-choice) ;
            vf   = decayV(vf, find([1:3]~=idx_color(choice)), decay) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
            vf           = update(vf, idxC, idxW, alpha_unrColor) ;
        elseif cntD==2
            idxW = idx_shape(choice) ;
            idxC = idx_shape(3-choice) ;
            vf   = decayV(vf, find([1:3]~=idx_shape(choice)), decay) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
            vf           = update(vf, idxC, idxW, alpha_unrShape) ;
        elseif cntD==3
            idxW = idx_pattern(choice) ;
            idxC = idx_pattern(3-choice) ;
            vf   = decayV(vf, find([1:3]~=idx_pattern(choice)), decay) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
            vf           = update(vf, idxC, idxW, alpha_unrPattern) ;
        end
    end
    if flag_couple
        if correctunCh
            if cntD==1
                idxC = idx_color(choiceunCh) ;
                idxW = idx_color(3-choiceunCh) ;
                [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
                vf = update(vf, idxC, idxW, alpha_rewColor) ;
        elseif cntD==2
                idxC = idx_shape(choiceunCh) ;
                idxW = idx_shape(3-choiceunCh) ;
                [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
                vf = update(vf, idxC, idxW, alpha_rewShape) ;
        elseif cntD==3
                idxC = idx_pattern(choiceunCh) ;
                idxW = idx_pattern(3-choiceunCh) ;
                [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
                vf = update(vf, idxC, idxW, alpha_rewPattern) ;
            end
        else
            if cntD==1
                idxW = idx_color(choiceunCh) ;
                idxC = idx_color(3-choiceunCh) ;
                [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
                vf = update(vf, idxC, idxW, alpha_unrColor) ;
            elseif cntD==2
                idxW = idx_shape(choiceunCh) ;
                idxC = idx_shape(3-choiceunCh) ;
                [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
                vf = update(vf, idxC, idxW, alpha_unrShape) ;
            elseif cntD==3
                idxW = idx_pattern(choiceunCh) ;
                idxC = idx_pattern(3-choiceunCh) ;
                [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
                vf = update(vf, idxC, idxW, alpha_unrPattern) ;
            end
        end
    end
end
end

% function v = decayV(v, unCh, decay)
%     v(unCh) = v(unCh)*(1-decay) ;
% end
function v = decayV(v, unCh, decay)
	v(unCh) = v(unCh) - (v(unCh)-0.5)*(decay) ;
end

function v = update(v, idxC, idxW, Q)
    if isempty(idxW)
        v(idxC) = v(idxC) + (1-v(idxC)).*Q ;
    elseif isempty(idxC)
        v(idxW) = v(idxW) - (v(idxW).*Q) ;
    elseif ~isempty(idxW) && ~isempty(idxC)
        v(idxC) = v(idxC) + (1-v(idxC)).*Q ;
        v(idxW) = v(idxW) - (v(idxW).*Q) ;
    end
end

function [idxW, idxC] = idxcouple(idxW, idxC, rl2_correct, flag_couple)
    if rl2_correct
        if flag_couple==0
            idxW = [] ;
        end
    else
        if flag_couple==0
            idxC = [] ;
        end
    end
end

function [idxW, idxC] = idxcoupleF(idxW, idxC, rl2_correct, flag_couple, flag_updatesim)
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