
function [snff,time_axis,freq_axis,best_t,fredges,tibins]=sniff_fielder16(spikes,locs,durs,window,bwidth,ifbin_edges)

Fs=1e3;
tibins=0:bwidth:window;
fredges=log2space(ifbin_edges(1),ifbin_edges(2),length(tibins));




% fredgex=fredgex(1:(end-1));
if length(fredges)>length(tibins)
    overhang=length(fredges)-length(tibins);
    fredges=fredges((overhang+1):length(fredges));
end
insfs=(Fs./durs);
spikemat=zeros(length(locs),1000)./0;


locs=locs(~isnan(insfs));
insfs=insfs(~isnan(insfs));
insfs(insfs>fredges(end))=fredges(end)-0.1;
insfs(insfs<fredges(1))=fredges(1)+0.1;


for this_loc=1:length(locs)
    loc_scan=spikes-locs(this_loc);
    loc_scan=loc_scan((loc_scan)<durs(this_loc)&loc_scan>=0);%durs(this_loc));
    spikemat(this_loc,1:length(loc_scan))=loc_scan;


end
snff=zeros(length(fredges)-1,length(tibins)-1);
limn=Fs./fredges(length(fredges):-1:1);
for this_freq=1:(length(fredges)-1)
    lineup=find(insfs>fredges(this_freq) & insfs<=fredges(this_freq+1));
    if length(lineup)>10
        this_batch=spikemat(lineup,:);
        this_batch=this_batch(~isnan(this_batch));
        batch_hist=histcounts(this_batch,tibins);
        
        sniff_row=batch_hist./length(lineup)./bwidth;



        

        
        snff(this_freq,:)=sniff_row;%[movmean(sniff_row,[0 5],2,"omitmissing" )];%smooth(sniff_row',3,'sgolay');
    end
end
snff=flipud(Fs.*snff);
if exist('imgaussfilt','file')
    snff=imgaussfilt( snff,6,'Padding','Symmetric','FilterDomain','spatial');
else
    snff=gauss_filt_symmetric(snff,6);%same filter, for MATLAB installs without the Image Processing Toolbox
end
% snff=meanfilt2d( snff,15, 15);

time_axis=zeros(1,size(snff,2));
% for m_s=1:size(snff,2)
%     if tibins(m_s)<=limn(1)
%         starter=1;
%     else
%         starter=ceil((tibins(m_s)-limn(1))/bwidth);
%     end
%     time_axis(m_s)=mean(snff(starter:end,m_s),1,"omitnan");
% 
% end


[~,best_t]=max(time_axis);
snff(isnan(snff))=0;
time_axis=max(snff,[],1);
freq_axis=max(snff,[],2);

tibins=tibins(1:(end-1));
fredges=fredges(1:(end-1));
end

function out=gauss_filt_symmetric(A,sigma)
%imgaussfilt(A,sigma,'Padding','symmetric','FilterDomain','spatial') in base MATLAB:
%normalized 2*ceil(2*sigma)+1 Gaussian kernel, edges padded by mirroring (edge sample included)
hw=ceil(2*sigma);
g=exp(-(-hw:hw).^2/(2*sigma^2));
g=g/sum(g);
ri=[hw:-1:1, 1:size(A,1), size(A,1):-1:size(A,1)-hw+1];
ci=[hw:-1:1, 1:size(A,2), size(A,2):-1:size(A,2)-hw+1];
out=conv2(g,g,A(ri,ci),'valid');
end
