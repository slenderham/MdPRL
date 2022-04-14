function PRL3v5(subName, flaginf) 
% reading images not making shapes
if nargin < 1
    subName = input('Initials of subject? [tmp]  ','s');                   % get subject's initials from user
end
if isempty(subName)
    subName = 'tmp';
end

if nargin < 2
    flaginf = str2double(input('Flag informative dimension? [1], 2  ','s'));
end
if isempty(flaginf) || ~(flaginf==1 || flaginf==2)
    flaginf = 1 ;
end
assert(exist(['./inputs/input_',subName,'.mat'], 'file')==0 | strcmp(subName,'tmp'), 'You cannot overwrite an existing file');

%% Generate input
fGenerateInputIndividual(subName, flaginf)

%%
load(['./inputs/input_',subName,'.mat']);
save(['./SubjectData/PRL_',subName,'.mat'],'expr');

%%
screenId = max(Screen('Screens'));
windowedMode = 0;
cnt_probe = 0 ;
try
    %% Set-up Screen
    AssertOpenGL;                                                          % Make sure this is running on OpenGL Psychtoolbox:
    InitializePsychSound(1);                                               % Low latency sound playback for making the beeps!
    Screen('Preference','SkipSyncTests', 1);
    Screen('Preference', 'SuppressAllWarnings', 1);
    PsychImaging('PrepareConfiguration');
    PsychImaging('AddTask', 'General', 'FloatingPoint32BitIfPossible');
    if ~windowedMode
        [w, rect] = PsychImaging('OpenWindow', screenId, 0);               % open screen
    else
        [w, rect] = PsychImaging('OpenWindow', screenId, 0, [0,0,800,600]);
    end
    ifi = Screen('GetFlipInterval', w);                                    % Retrieve video redraw interval for later control of our animation timing:
    AssertGLSL;                                                            % Make sure the GLSL shading language is supported:
    Screen('BlendFunction', w, GL_ONE, GL_ONE);
    Screen('TextFont', w, 'Arial');
    Screen('TextSize', w, 25);
    KbReleaseWait;
    
    %%
    setup.res = rect(3:4)-rect(1:2);                                             % screen resolution
    setup.x0 = setup.res(1)/2;
    setup.y0 = setup.res(2)/2;
    
    fixCoords(:,1) =  [setup.x0-setup.bigSize,setup.y0-setup.smallSize,setup.x0+setup.bigSize,setup.y0+setup.smallSize] ; 
    fixCoords(:,2) =  [setup.x0-setup.smallSize,setup.y0-setup.bigSize,setup.x0+setup.smallSize,setup.y0+setup.bigSize] ;
    
    targRectN(1,:)=CenterRectOnPoint([0 0 setup.targSizeN setup.targSizeN],setup.x0-setup.targDist,setup.y0);  % left side
    targRectN(2,:)=CenterRectOnPoint([0 0 setup.targSizeN setup.targSizeN],setup.x0+setup.targDist,setup.y0);  % right side
    
    totalreward  = 0  ;
    runningreward = 0 ;
    
    %% Begin Experiment
    DrawFormattedText(w,'Press G to begin','center','center',256);
    vbl = Screen('Flip', w);
    FlushEvents('keyDown');
    [~, ~, keyCode] = KbCheck;
    while ~keyCode(setup.gKey)
        [~, ~, keyCode] = KbCheck;
    end
    for trNo = 1:expr.Ntrials
        
        if trNo<=expr.Ntrials
            for cnt_shape = 1:3
                for cnt_texture = 1:3
                    for cnt_color = 1:3
                        color = expr.colorList(expr.color(cnt_color),:) ;
                        image{cnt_shape,cnt_texture, cnt_color} = imread(sprintf('%s%s%s%s','./Symbols/symbol_',int2str(expr.shape(cnt_shape)), int2str(expr.texture(cnt_texture)),'.jpg')) ;
                        for cntRGB = 1:3
                            image{cnt_shape,cnt_texture, cnt_color}(:,:,cntRGB) = image{cnt_shape,cnt_texture, cnt_color}(:,:,cntRGB)*color(cntRGB) ;
                        end
                    end
                end
            end
        end
        if trNo==1
            WaitSecs(0.1)
            FlushEvents('keyDown');
            keyCode=zeros([1 256]);
            fPlotStim(w, image, expr, setup)
            DrawFormattedText(w,'Stimuli set',setup.x0+setup.banCoords(1),setup.y0-2.5*setup.banCoords(2),256);
            DrawFormattedText(w,'Press G to play',setup.x0+setup.banCoords(1),setup.y0+2.5*setup.banCoords(2),256);
            vbl = Screen('Flip', w);
            FlushEvents('keyDown');
            [~, ~, keyCode] = KbCheck;
            while ~keyCode(setup.gKey)
                [~, ~, keyCode] = KbCheck;
            end
        end
        
        [imgWidthL, imgHeightL, temp] = size(image{input.inputTarget(1, trNo)});
        [imgWidthR, imgHeightR, temp] = size(image{input.inputTarget(2, trNo)});
        
        targL = Screen('MakeTexture',w,image{input.inputTarget(1, trNo)});
        targR = Screen('MakeTexture',w,image{input.inputTarget(2, trNo)});
        
        CoordL = [setup.x0-setup.targDist-imgWidthL/2 setup.y0-imgHeightL/2 setup.x0-setup.targDist+imgWidthL/2 setup.y0+imgHeightL/2] ;
        CoordR = [setup.x0+setup.targDist-imgWidthR/2 setup.y0-imgHeightR/2 setup.x0+setup.targDist+imgWidthR/2 setup.y0+imgHeightR/2] ;
        
        rep = 1 ;
        flagmessage = 0 ;
        RewBannerOn = 0;
        while rep ~= 0 
            FlushEvents('keyDown');
            keyCode=zeros([1 256]);
            
            % ITI: Dram fixation point for random time U(0.5, 1.5)
            Screen('FillRect',w,0);
%             Screen('FillRect',w,setup.centerFeedback,fixCoords);
            if setup.feedbackOn == 0 
            else
                Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
            end
            vbl = Screen('Flip', w);
            waitTime(1,trNo) = setup.beforeWait(1)+rand*diff(setup.beforeWait);
            WaitSecs(waitTime(1,trNo));
            
            % Fixation time: Dram fixation point
            Screen('FillRect',w,0);
            Screen('FillRect',w,setup.correctFeedback,fixCoords);
            if setup.feedbackOn == 0 
            else
                Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
            end
            vbl = Screen('Flip', w);
            WaitSecs(setup.fix);
            
            % Draw targets
            Screen('FillRect',w,0);
%             Screen('FillRect',w,setup.correctFeedback,fixCoords);
            Screen('FillRect',w,setup.centerFeedback,fixCoords);
            
            Screen('DrawTexture',w,targL,[],CoordL) ;
            Screen('DrawTexture',w,targR,[],CoordR) ;
            
            if setup.feedbackOn == 0 
            else
                Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
            end
            vbl = Screen('Flip', w);
            tStart = vbl ;
            tpresent = tStart ;                                            % initialize the RT/trial duartion timer
            if rep==1
               tdispDur = setup.dispDur ;
            else
               tdispDur = setup.dispDur/2 ;
            end
            while tpresent<tStart+tdispDur+setup.extrDsc && sum(keyCode([setup.leftKey,setup.rightKey]))== 0
                [~, ~, keyCode] = KbCheck;                                 % Look for response
                if tpresent<tStart+setup.dispDur
                    Screen('FillRect',w,0);
                    Screen('FillRect',w,setup.correctFeedback,fixCoords);
                    
                    Screen('DrawTexture',w,targL,[],CoordL) ;
                    Screen('DrawTexture',w,targR,[],CoordR) ;
                    if rep==1 & tpresent>tStart+setup.dispDur/2 & tpresent<tStart+setup.dispDur
                        flagmessage = 1 ;
                    end
                    
                    if setup.feedbackOn == 0 
                    else
                        Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
                    end
                    if flagmessage
                        Screen('DrawText',w,'Select side please!',setup.x0+setup.banCoords(1),setup.y0-setup.banCoords(2),setup.angrymessageColor);
                    end
                    vbl = Screen('Flip', w);
                else
                    Screen('FillRect',w,0);
                    Screen('FillRect',w,setup.correctFeedback,fixCoords);
                    if setup.feedbackOn == 0 
                    else
                        Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
                    end
                    vbl = Screen('Flip', w);
                end
                tpresent = vbl ;
            end
            if sum(keyCode([setup.leftKey,setup.rightKey]))== 0
                rep = rep + 1 ;
                flagmessage = 1 ;
            else
                responsetime(trNo,1) = GetSecs - tStart ;
                if keyCode(setup.leftKey)
                    choice(trNo,1) = 1 ;
                    repeat(trNo,1) = rep ;
                else
                    choice(trNo,1) = 2;
                    repeat(trNo,1) = rep ;
                end
                rep = 0 ;
            end
        end
        %% present choice
        tStart = GetSecs;
        tpresent = tStart ;
        while tpresent<tStart+setup.choicePresentationT
            Screen('FillRect',w,0);
            Screen('FillRect',w,setup.correctFeedback,fixCoords);
            Screen('FrameOval', w, setup.neutralFeedback, targRectN(choice(trNo,1),:), 5);
            
            Screen('DrawTexture',w,targL,[],CoordL) ;
            Screen('DrawTexture',w,targR,[],CoordR) ;
            
            vbl = Screen('Flip', w);
            tpresent = vbl ;
        end
        
        %% present reward feedback
        
        reward(trNo,1) = input.inputReward(choice(trNo), trNo) ;
        reward(trNo,2) = input.inputReward(3-choice(trNo), trNo) ;
        if reward(trNo,1)
            feedbackColor = setup.correctFeedback;
            totalreward = totalreward + 1 ;
            runningreward = runningreward + 1 ;
        else
            feedbackColor = setup.wrongFeedback;
            totalreward = totalreward ;
            runningreward = runningreward ;
        end
        if reward(trNo,2)
            feedbackColorNotchosen = setup.correctFeedback ;
        else
            feedbackColorNotchosen = setup.wrongFeedback ;
        end
        if setup.feedbackOn == 0 
            tStart = GetSecs;
            tpresent = tStart ;
            rewardPresentationTemp = setup.rewardPresentationT ; 
            while tpresent<tStart+rewardPresentationTemp
                if RewBannerOn == 0
                    Screen('FrameOval', w, setup.neutralFeedback, targRectN(choice(trNo,1),:), 5);
                end
                if tpresent>tStart+setup.overallrewfeedbacktime
                    if  runningreward>= setup.rewardpoint2Money && RewBannerOn == 0
                        Screen('DrawText',w,'You earn 25 cents',setup.x0+banCoords(1),setup.y0+setup.banCoords(2),setup.messageColor);
                        runningreward = mod(runningreward,setup.rewardpoint2Money) ;
                        Screen('Flip',w);
                        RewBannerOn = 1 ;
                        rewardPresentationTemp = rewardPresentationTemp + setup.BannerDisTime ;
                    elseif RewBannerOn == 1
                        Screen('DrawText',w,'You earn 25 cents',setup.x0+setup.banCoords(1),setup.y0+setup.banCoords(2),setup.messageColor);
                    end
                end
                Screen('FillRect',w,setup.centerFeedback,fixCoords);
                
                Screen('DrawTexture',w,targL,[],CoordL) ;
                Screen('DrawTexture',w,targR,[],CoordR) ;
                
                vbl = Screen('Flip', w, vbl + 0.5 * ifi);
                tpresent = vbl ;
            end
        else
            tStart = GetSecs;
            tpresent = tStart ;
            rewardPresentationTemp = setup.rewardPresentationT ;
            while tpresent<tStart+rewardPresentationTemp
                Screen('FrameOval', w, feedbackColor, targRectN(choice(trNo,1),:), 5);
                if setup.feedbackNotchosen==1
%                     Screen('FrameOval', w, feedbackColorNotchosen, targRectN(3-choice(trNo,1),:), 5);
                end
                Screen('FillRect',w,setup.correctFeedback,fixCoords);

                if reward(trNo,1)
                    Screen('DrawText',w,num2str(1),setup.x0+(-1)^choice(trNo,1)*setup.targDist-6,setup.y0-setup.targSizeN/1.5-10,setup.rewardColor);
                else
                    Screen('DrawText',w,num2str(0),setup.x0+(-1)^choice(trNo,1)*setup.targDist-6,setup.y0-setup.targSizeN/1.5-10,setup.punishColor);
                end
                if setup.feedbackNotchosen==1 & tpresent>tStart+0.5
                    if reward(trNo,2)
                        Screen('DrawText',w,num2str(1),setup.x0+(-1)^(3-choice(trNo,1))*setup.targDist-6,setup.y0-setup.targSizeN/1.5-10,setup.rewardColor);
                    else
                        Screen('DrawText',w,num2str(0),setup.x0+(-1)^(3-choice(trNo,1))*setup.targDist-6,setup.y0-setup.targSizeN/1.5-10,setup.punishColor);
                    end
                end
                
                if tpresent>tStart+setup.overallrewfeedbacktime
                    if  runningreward >= setup.rewardpoint2Money || RewBannerOn  == 1
                        if RewBannerOn == 0
                            Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
                            runningreward = mod(runningreward,setup.rewardpoint2Money) ;
                            Screen('Flip',w);
                            RewBannerOn = 1 ;
                            rewardPresentationTemp = rewardPresentationTemp + setup.BannerDisTime ;
                        else
                            Screen('DrawText',w,'You earn 25 cents',setup.x0+setup.banCoords(1),setup.y0+setup.banCoords(2),setup.messageColor);
                        end
                    else
                        Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
                    end
                end
                
                Screen('DrawTexture',w,targL,[],CoordL) ;
                Screen('DrawTexture',w,targR,[],CoordR) ;
                
                vbl = Screen('Flip', w, vbl + 0.5 * ifi);
                tpresent = vbl ;
            end
        end
        %% compensate for RT
        tStart = GetSecs;
        tpresent = tStart ;
        while tpresent<tStart+(3-responsetime(trNo,1) - (waitTime(trNo)-1) )
            Screen('FillRect',w,0);
%             Screen('FillRect',w,setup.centerFeedback,fixCoords);
            if setup.feedbackOn == 0 
            else
                Screen('DrawText',w, sprintf('%s%d','Reward points = ', runningreward),setup.x0+setup.rewCoords(1),setup.y0+setup.rewCoords(2),setup.totalColor);
            end
            vbl = Screen('Flip', w);
            tpresent = vbl ;
        end
        
        %%
        results.choice = choice ;                                          % subjects choice (1-left, 2-right)
        results.reward = reward(:,1) ;                                     % reward responses
        results.waitTime = waitTime ;
        results.responsetime = responsetime ;
        results.reapeat = repeat ;
        
        %%
        if ismember(trNo, expr.trialProbe)
            cnt_probe = cnt_probe + 1 ;
            [responseTime{cnt_probe}, probEst{cnt_probe}] = fDrawStimProbe(setup, image, w, expr.playcombinations) ;
            results.responseTime{cnt_probe} = responseTime{cnt_probe} ;
            results.probEst{cnt_probe} = probEst{cnt_probe} ;
        end
        save(['./SubjectData/PRL_',subName,'.mat'],'results','-append');
        %% pause in the middle
        if trNo==expr.Ntrials/2
            pauseOn = 1 ;
        else
            pauseOn = 0 ;
        end
        if pauseOn
            DrawFormattedText(w,'When ready press G to resume',setup.x0-150,setup.y0+50, 256);
            Screen('Flip', w);
            while pauseOn
                [~, ~, keyCode] = KbCheck;
                if keyCode(setup.gKey)
                    pauseOn=0;
                end
            end
        end
    end
    %% Finish Experiment
    sca;
    ListenChar(1);
    ShowCursor;
    
catch err
    ListenChar(1);
    
    save('error.mat');                                                     % save the whole workspace to work on error
    disp('): something went wrong :(');
    disp(' ');
    rethrow(err);
end
