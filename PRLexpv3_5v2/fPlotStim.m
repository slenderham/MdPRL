function fPlotStim(w, image, expr, setup)
cnt_image = 0 ;
y0(1) = setup.y0+1.4*setup.banCoords(2) ;
y0(2) = setup.y0-.0*setup.banCoords(2) ;
y0(3) = setup.y0-1.4*setup.banCoords(2) ;
qoutientj = ceil(length(expr.playcombinations)/3) ;

D = setup.res(1)./(qoutientj+4) ;
imageLoc = D.*([1:qoutientj]-(qoutientj+1)/2) ;
imageLoc_shift = D.*([1:qoutientj]-(qoutientj)/2) ;

for idximage = sort(expr.playcombinations)
    cnt_image = cnt_image + 1 ;
    j = ceil(cnt_image/qoutientj) ;
    i = mod(cnt_image, qoutientj) ;
    if i==0
        i = qoutientj ;
    end
    if mod(length(expr.playcombinations),qoutientj)~=0 & j==2
        targDist = imageLoc_shift ;
    else
        targDist = imageLoc ;
    end
    [imgWidthL, imgHeightL, temp] = size(image{idximage});
    Coord = [setup.x0-targDist(i)-imgWidthL/2 y0(j)-imgHeightL/2 setup.x0-targDist(i)+imgWidthL/2 y0(j)+imgHeightL/2] ;
    targ = Screen('MakeTexture',w,image{idximage});
    Screen('DrawTexture',w,targ,[],Coord) ;
end