function hline = plot_shaded_errorbar(m, sd, wSize, clr)
    if size(wSize)==1
        uX = wSize:432;
    else
        uX = wSize;
    end
    x = [uX, fliplr(uX)];
    y = [m'+sd', fliplr(m'-sd')];
    hold on
    hpatch    = patch(x,y,'b');
    set(hpatch,'EdgeColor','none');
    set(hpatch,'FaceColor',clr);
    hline     = plot(uX,m,'-','Color',clr);
    set(hline,'LineWidth',1.5);
    set(hline,'Color',clr);
    box off
    alpha(hpatch,0.2);
end