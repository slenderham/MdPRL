function [responseTime, probEst] = fDrawStimProbe(setup, image, w, playcombinations)

targetlist = playcombinations ;
responseTime = nan*ones(3,3,3) ;
probEst = nan*ones(3,3,3) ;

for cnt_target = targetlist(randperm(length(targetlist)))
    WaitSecs(0.25)
    Screen('FillRect',w,0);
    [imgWidth, imgHeight, temp] = size(image{cnt_target});
    sym_myPhotoTex = Screen('MakeTexture',w,image{cnt_target});
    CoordM = [setup.x0-imgWidth/2 setup.y0-imgHeight/2 setup.x0+imgWidth/2 setup.y0+imgHeight/2] ;
    Screen('DrawTexture',w,sym_myPhotoTex,[],CoordM) ;
    
    % Draw probability estimate circles  
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-4.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-3.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-2.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-1.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-0.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+0.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+1.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+2.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+3.5*setup.D,setup.y0+setup.D));
    Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+4.5*setup.D,setup.y0+setup.D));
    Screen(w,'TextSize',setup.textFontSize*0.8);
    
    centerText(w,'<10%',setup.x0-4.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'10-20%',setup.x0-3.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'20-30%',setup.x0-2.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'30-40%',setup.x0-1.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'40-50%',setup.x0-.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'50-60%',setup.x0+.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'60-70%',setup.x0+1.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'70-80%',setup.x0+2.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'80-90%',setup.x0+3.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    centerText(w,'>90%',setup.x0+4.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
    Screen(w,'TextSize',setup.textFontSize);
    vbl = Screen('Flip', w);
    
    tStart = vbl ;
    rp = 0 ;
    FlushEvents('keyDown');
    keyCode=zeros([1 256]);
    while rp~=1
        [~, ~, keyCode] = KbCheck ;
        if (sum(keyCode([setup.Probekeys]))~=0)
            responseTime(cnt_target) = GetSecs-tStart ;
            key = find(keyCode([setup.Probekeys])) ;
            key = key(1) ;
%             probEst(cnt_target) = setup.Prob(key) ;
            probEst(cnt_target) = setup.LL(key) ;
            rp = 0.5 ;
            
            Screen('FillRect',w,0);
            Screen('DrawTexture',w,sym_myPhotoTex,[],CoordM) ;
            % Draw probability estimate circles  
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-4.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-3.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-2.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-1.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0-0.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+0.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+1.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+2.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+3.5*setup.D,setup.y0+setup.D));
            Screen('FrameOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+4.5*setup.D,setup.y0+setup.D));
            Screen(w,'TextSize',setup.textFontSize*0.8);

            centerText(w,'<10%',setup.x0-4.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'10-20%',setup.x0-3.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'20-30%',setup.x0-2.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'30-40%',setup.x0-1.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'40-50%',setup.x0-.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'50-60%',setup.x0+.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'60-70%',setup.x0+1.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'70-80%',setup.x0+2.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'80-90%',setup.x0+3.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            centerText(w,'>90%',setup.x0+4.5*setup.D,setup.y0+setup.D+setup.fixRectEdge(3)+10,setup.textColor);
            Screen(w,'TextSize',setup.textFontSize);
            % Draw choice
            Screen('FillOval',w,setup.textColor,CenterRectOnPoint(setup.fixRectEdge,setup.x0+setup.Dn(key)*setup.D,setup.y0+setup.D));
            vbl = Screen('Flip', w);
            
            FlushEvents('keyDown');
            keyCode=zeros([1 256]);
        end
        if (sum(keyCode([setup.Enterkey]))~=0)
            rp = 1 ;
        end
        
    end
    
end

end