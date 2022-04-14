function [setup] = fPRL2initSetup

setup.feedbackOn = 1 ;                                 % turn feedback on or off
setup.feedbackNotchosen = 1 ;
setup.va2p = 75;                                       % pixels per visual degree

setup.neutralFeedback = [255 255 255] ;
setup.correctFeedback = [255 255 -128]; 
setup.wrongFeedback =   [255/2 255/2 255/2] ;
setup.totalColor =      [255 255 -128] ;
setup.messageColor =    [255 255 -128] ;
setup.angrymessageColor = [255 -128 255] ;
setup.textColor = [255 255 255] ;
setup.centerFeedback = setup.neutralFeedback ;

setup.rewardColor = setup.correctFeedback;
setup.punishColor = setup.wrongFeedback;

setup.choicePresentationT = .5 ;
setup.rewardPresentationT = 1.0 ;
setup.dispDur = 5 ; %2.5 ;                           % how long are stimuli displayed for first round of each trial, for second round half of this time
setup.extrDsc = .5 ;                                 % how long extra descion time after stimuli is removed
setup.targSize = round(setup.va2p*.8) ;              % size of target
setup.fix = .75 ;
setup.targSizeN = round(setup.va2p*2.2) ;             % size of choice/reward circle
setup.targDist = round(setup.va2p*4) ;               % target distance from center
setup.beforeWait = [0.25 0.75];

setup.overallrewfeedbacktime = 0.7 ;                 % after which it comes on
setup.BannerDisTime = 1.5 ;
setup.rewardpoint2Money = 10 ;
setup.rewardFBwhenNoFB = 5 ;                         % interval in which to show reward FB when FB is off

% targets parameter
setup.Color(1,:) = [255 -128 -128] ;
setup.Color(2,:) = [-128 -128 255] ;
setup.Color(3,:) = [255 -128 255] ;

% keyboard setup
KbName('UnifyKeyNames') ;
setup.leftKey = KbName('LeftArrow') ;                % left choice
setup.rightKey = KbName('RightArrow') ;              % right choice
setup.gKey = KbName('g') ;                           % go key 

% fixation cross setup
setup.bigSize = 15 ;
setup.smallSize = 5 ;

% reward feedback location
setup.rewCoords = [-90 180] ;
setup.banCoords = [-90 120] ;

setup.Probekeys(1) = KbName('a');
setup.Probekeys(2) = KbName('s');
setup.Probekeys(3) = KbName('d');
setup.Probekeys(4) = KbName('f');
setup.Probekeys(5) = KbName('g');
setup.Probekeys(6) = KbName('h');
setup.Probekeys(7) = KbName('j');
setup.Probekeys(8) = KbName('k');
setup.Probekeys(9) = KbName('l');
if ismac
    setup.Probekeys(10) = KbName(';:');
elseif ispc
    setup.Probekeys(10) = KbName(';');
end
setup.Enterkey = KbName('Return');

setup.Prob = [5 15 25 35 45 55 65 75 85 95]/100 ;
setup.LL = setup.Prob./(1-setup.Prob) ;
setup.Dn = [-4.5 -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5] ;
setup.fixRectEdge = [0 0 20 20];
setup.D = 120 ;
setup.textFontSize = 25;
    
end

