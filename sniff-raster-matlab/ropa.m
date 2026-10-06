function mc=ropa


mc=(colorcet('R2'));
mult=linspace(0,1,size(mc,1))';
mult=mult./max(mult);
mult=[mult mult mult];
mc=fliplr(1-mc.*mult);
% colormap(mc);