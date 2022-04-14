function centerText(theWindow,text,x,y,color)

% center text on (x,y)
%
% SWU.

[nBRect,oBRect]= Screen('TextBounds',theWindow,text,x,y,[]);

w = nBRect(3);
% The drawText function places text vertically on y.
% To change vertical displacement so that vertical center of the text is the center
% of the word, we do the following:
h = nBRect(4);   %font size.

% Screen(theWindow,'DrawText',text,x-w/2,y+h/2,color);
Screen(theWindow,'DrawText',text,x-w/2,y,color);

