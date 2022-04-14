function CalibrateEyelinkRect(window,rect)

global abbreviatedFilename
%filename = 'test'%%%abbreviatedFilename;
% 
%eyelinkScreenSize = [0 0 500 500];
eyelinkScreenSize = [0 0 1600 1200];

KbName('UnifyKeyNames');

matlabScreenSize = rect;
smallWindow = CenterRect(eyelinkScreenSize, matlabScreenSize);
spacebar=KbName('space');
% rightkey=KbName('Right');
% leftkey=KbName('Left');
leftKey=KbName('LeftArrow');     % left choice
rightKey=KbName('RightArrow');   % right choice
upkey=KbName('Return');
upkey =upkey(1) ;
backspacekey = KbName('DELETE');
% upkey=KbName('Return');
% backspacekey = KbName('BackSpace');

Eyelink('Shutdown');
Eyelink('Initialize');


%window = Screen(0,'OpenWindow');%, [], smallWindow);
HideCursor
[sFactorOld, dFactorOld]=Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); % temporarily set the blend function to standard, 
                                                                  %  and record previous blend function
Screen('Fillrect', window, [255 255 255])
Screen('Flip',window);

calibAndValidate=0;

Eyelink('StartSetup')
pause(2)
% result = Eyelink('CalResult');
%         if result == 0
%             calibAndValidate=calibAndValidate+1;
%         else
%         end

% [KeyIsDown,Secs,keyCode] = KbCheck;
% while keyCode(13) ~= 1

whichKey=0;

keysWanted=[spacebar upkey backspacekey];
                FlushEvents('keydown');
                success = 0;
                while success == 0
                    pressed = 0;
                    while pressed == 0
                        [pressed, keyTime, kbData] = KbCheck;
                    end;
                    
                    for keysToCheck = 1:length(keysWanted)
                        if kbData(keysWanted(keysToCheck)) == 1
                            
                            
                            keyPressed = keysWanted(keysToCheck);
                            if keyPressed == backspacekey
                                whichKey=9;
                                FlushEvents('keydown');
                                WaitSecs(.1)
                            elseif keyPressed == spacebar
                                whichKey=1;
                            FlushEvents('keydown');
                            WaitSecs(.1)
                            elseif keyPressed == upkey
                                whichKey=5;
                                FlushEvents('keydown');
                                WaitSecs(.1)
                            else
                            end
                            FlushEvents('keydown');
                            keyPressed=0;
                            
%                             break;
                        end;
                    end;
%                     FlushEvents('keydown');
                    
                    if whichKey == 1
                        whichKey=0;
                        [visible, tx, ty] = Eyelink('TargetCheck');
% %                      tx =tx*(1024/1600)+1024;%added for 3 screen setup adjustment
% %                      ty =ty*(768/1200);
            Screen('FillRect', window ,[0 0 0], [tx-20 ty-5 tx+20 ty+5]);
            Screen('FillRect', window ,[0 0 0], [tx-5 ty-20 tx+5 ty+20]);
            Screen('Flip', window);
            elseif whichKey == 5
                whichKey=0;
             Eyelink('AcceptTrigger');
                    elseif whichKey == 9
                        break
                    else
                    end
%                     success = 1;
                end;
%         if whichKey == 1
%             break
%         else
%         end

% calibrationComplete = 0;
% 
%     while calibrationComplete ~=1 %calibAndValidate ~= 2
%         WaitSecs(.2)
%         [visible, tx, ty] = Eyelink('TargetCheck');
%         while visible == 0
%         [visible, tx, ty] = Eyelink('TargetCheck');
%         keysWanted=[backspacekey];
%         [pressed, keyTime, kbData] = KbCheck;
%         for keysToCheck = 1:length(keysWanted)
%         keyPressed = keysWanted(keysToCheck);
%         end
%                             if keyPressed == backspacekey break
%                             else
%                             end
%          WaitSecs(.1)
%         end
%         
%         tx =tx+1024;%added for 3 screen setup adjustment
%         
%             Screen('FillRect', window ,[0 0 0], [tx-20 ty-5 tx+20 ty+5]);
%             Screen('FillRect', window ,[0 0 0], [tx-5 ty-20 tx+5 ty+20]);
%             Screen('Flip', window);
% 
%              keysWanted=[spacebar upkey];
%                 FlushEvents('keydown');
%                 success = 0;
%                 while success == 0
%                     pressed = 0;
%                     while pressed == 0
%                         [pressed, keyTime, kbData] = KbCheck;
%                     end;
%                     
%                     for keysToCheck = 1:length(keysWanted)
%                         if kbData(keysWanted(keysToCheck)) == 1
%                             
%                             success = 1;
%                             keyPressed = keysWanted(keysToCheck);
%                             if keyPressed == backspacekey
%                                 whichKey=1;
%                             elseif keyPressed == upkey
%                                 whichKey=5;
%                             end
%                             FlushEvents('keydown');
%                             break;
%                         end;
%                     end;
%                     FlushEvents('keydown');
%                 end;
%         if whichKey == 1
%             calibrationComplete = 1;
%         elseif whichKey == 5
%              success = Eyelink('AcceptTrigger');
% %             break
%         else
%         end
%            
%             
%           
%         %         else
%         %         end
%         
%         %     end
% %         result = Eyelink('CalResult');
% %         if result == 0
% %             calibAndValidate=calibAndValidate+1;
% %         else
% %         end
%         
%         
%     end

Screen('Fillrect', window, 128);    % DL: go back to old background (fixes 'inverted' gabors)

Screen('BlendFunction', window, sFactorOld, dFactorOld); % reset the blend function
Eyelink('OpenFile',abbreviatedFilename)
Eyelink('StartRecording')
%sca