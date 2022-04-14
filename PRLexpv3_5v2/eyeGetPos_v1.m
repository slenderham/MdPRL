function [trueEyePos,tempDist,rawEyePos] = eyeGetPos_v1(x0,y0,eye_used,debugMode)
    if debugMode % use mouse position, rather than eye position
        [eyePos.gx(1),eyePos.gy(1)] = GetMouse;
    else
        eyePos = Eyelink('NewestFloatSample');
    end
             
    xPos = eyePos.gx(eye_used+1)-x0;
    yPos = eyePos.gy(eye_used+1)-y0;
    rawEyePos = [eyePos.gx(eye_used+1),eyePos.gy(eye_used+1)];
    trueEyePos=[xPos,yPos];
    tempDist = sqrt(xPos^2+yPos^2);
end

