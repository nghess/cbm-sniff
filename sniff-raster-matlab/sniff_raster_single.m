function out = sniff_raster_single(session_dir, region, cl, out_dir)
% SNIFF_RASTER_SINGLE  Sniff field + raster plots for one cluster in one session.
%
%   out = sniff_raster_single(session_dir, region, cl, out_dir)
%
%   session_dir : session folder (path ends in .../<mouse_id>/<date>)
%                 containing sniff_params.mat and the ob-ks/ and hc-ks/
%                 sorting subfolders
%   region      : 'ob' (olfactory bulb) or 'hc' (CA1); selects the
%                 <region>-ks/ subfolder holding spike_times.npy and
%                 spike_clusters.npy. Cluster ids are only unique within a region.
%   cl          : cluster id (a value from spike_clusters.npy, not an index)
%   out_dir     : where figures and the reference .mat are written
%                 (default: ./reference_output)
%
%   Same computation as sniff_raster.m for a single cluster, except that:
%     - figures are always made (no "raster png already exists" skip)
%     - the unit quality gate is reported but does not stop the run
%     - every intermediate is returned in `out` and saved to
%       <out_dir>/<titl>_reference.mat (v7, readable by scipy.io.loadmat)
%       so a Python port can be checked against it step by step

region = validatestring(region, {'ob', 'hc'});
if nargin < 4 || isempty(out_dir)
    out_dir = fullfile(pwd, 'reference_output');
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end

pmax=250;
pmin=-pmax;
bwidth=2;
fredge_edges=[2 13];
%refractory period violation cutoff
vio=0.05;
% set whether it's chronologically or sniff dur sorted
chron=0;%0=dur sort 1=chronological
%align to i_locs or e_locs
inl=1; %1 = inhalation ; 0 = exhalation
Fs=1e3;
load('Riley_Shadowplay.mat', 'ropa')
%set time axis limits
maxax=1e4;

txtsize=20;

sp=strsplit(session_dir, {'/', '\'});
sp=sp(~cellfun(@isempty, sp));
mouse_id=sp{end-1};
date=sp{end};

ks_dir=fullfile(session_dir, [region '-ks']);

spike_times=readNPY(fullfile(ks_dir, 'spike_times.npy'));
clusters=readNPY(fullfile(ks_dir, 'spike_clusters.npy'));
load(fullfile(session_dir, 'sniff_params.mat'),'sniff_params');
clinclude=flipud(unique(clusters));

% position of this cluster in the original loop order; only the first
% cluster (cll==1) gets x tick labels on the time axis
cll=find(clinclude==cl);
if isempty(cll)
    error('Cluster %d not found in %s', cl, ks_dir)
end

i_locs=sniff_params(:,1);
e_locs=sniff_params(:,3);

% this code strips out the bad inhalation times
b_locs=i_locs;
b_locs(e_locs==0)=nan;

durs=diff(b_locs);

i_locs=i_locs(~isnan(durs));
e_locs=e_locs(~isnan(durs));
durs=durs(~isnan(durs));
if inl==1
    locs=i_locs;
elseif inl==0
    locs=e_locs;
end

durs=durs(locs<spike_times(end)/30);
locs=locs(locs<spike_times(end)/30);

titl=[mouse_id ' ' date ' ' region ' cl ' num2str((cl))];

spikes=(double(spike_times(clusters==cl ))./30);

interv=diff(spikes);

rpv=length(find(interv<1.5))/length(interv);

over=Fs*length(spikes)/range(spike_times./30);

if length(spikes)<50
    rpv=1;
    over=0;
end

passes_gate = over>0.1 &&  over<66 && rpv<vio;
fprintf('%s: rate=%.3f Hz, rpv=%.4f, passes quality gate: %d\n', titl, over, rpv, passes_gate)
if ~passes_gate
    warning('sniff_raster.m would have skipped this cluster.')
end
if isempty(locs)
    error('No usable sniffs in %s', session_dir)
end

%% sniff field plot

[snff,stime_axis,sfreq_axis,~,fredges,xax]=sniff_fielder16(spikes,locs,durs,pmax,bwidth,fredge_edges);
snff_raw=snff;

limn=(Fs./fredges(1:(end)))./bwidth;
limn_raw=limn;

limn=smooth(limn',11,'sgolay')';
limn_smooth=limn;
limn=round(fliplr(limn))';
limn(limn>size(snff,2))=size(snff,2);

sps_lim=[floor(min([stime_axis'; sfreq_axis],[],"all")) ceil(max([stime_axis'; sfreq_axis],[],"all")) ];

if max(sps_lim)==0
    sps_lim=[0 1];
end
for m=1:size(snff,1)
    snff(m,limn(m):end)=0;
end

figure('Color','w')
snf_ax=subplot(2,2,2);

imagesc((snff))
hold on

colormap(ropa);
clim(sps_lim)
lw=3;
%limn the snff
plot(limn,1:length(limn),'k-','LineWidth',lw)

plot(xlim,[size(snff,1) size(snff,1)],'k-','LineWidth',lw)
plot([1 1],ylim,'k-','LineWidth',lw)
plot([1 limn(1)],[1 1],'k-','LineWidth',lw)

plot([size(snff,2) size(snff,2)],[size(snff,1) limn(size(snff,2)) ],'k-','LineWidth',lw)

yticks([1 size(snff,1)])
yticklabels([12 2])

box off
hcb = colorbar;
hcb.LineWidth=1;
hcb.Label.String = [ 'Spikes per s'];

hcb.Label.Rotation = -90;

hcb.Ticks=sps_lim;

axis off
xlabel('Latency from inhalation (ms)')

yt(txtsize)

tim_ax = subplot(2,2,4);

hold on

plot(xax,stime_axis,'color','k','linewidth',2)

if cll==1
    %couldn't figure out how to position these in
    %code so gotta do it in illustrator
    xticks([0:100:pmax])
    xlabel('Latency from inhalation (ms)')
else
    xticks([])
end

ylabel('Spikes per s');

xlim([0 pmax])
yt(txtsize)
ylim(sps_lim)
yticks(sps_lim)

freq_ax = subplot(2,2,1);

hold on

plot(flipud(sfreq_axis),fredges,'color','k','linewidth',2)

ylim([fredges(1) 12])
set(freq_ax,'Yscale','log')
set(freq_ax,'XAxisLocation','top')
xlim(sps_lim)
xticks(sps_lim)

xlabel(freq_ax,'Spikes per s');

yt(txtsize)

set(snf_ax,'Position', [ 0.1 0.09 0.66 0.66])
hcb.Position= [0.5683    0.3888    0.0500    0.3060];
set(tim_ax,'Position', [ 0.1 0.8 0.66 0.17])
set(freq_ax,'Position', [ 0.8 0.09 0.17 0.66])
set(snf_ax,'LineWidth',2)
set(tim_ax,'LineWidth',2)
set(freq_ax,'LineWidth',2)
set(gcf,'Position',[200 200 720 720])

pause(2)
print(gcf, '-dsvg',fullfile(out_dir, [titl '_SnF.svg']))

%% raster plot

raster_stack=[];

if length(locs)>maxax
    breath_ind=round(linspace(1,length(locs), maxax));
else
    breath_ind=round(linspace(1,length(locs), length(locs)));
end

if chron==0
    [~,ord]=sort(durs,'ascend');
else
    ord=1:length(durs);
end

for bre=1:length(breath_ind)
    %can sort by Latency from next or previous
    %inhalation/exhalation
    breath=breath_ind(bre);
    next= durs(ord(breath));
    i_f=Fs./next;

    shift_spikes=spikes-locs(ord(breath));

    eligible=shift_spikes(shift_spikes>(pmin) & shift_spikes<(pmax));

    counts=ones(length(eligible),1);
    inds=bre.*counts;
    ifs=i_f.*counts;
    nexts=next.*counts;

    raster_stack=[raster_stack; inds eligible  nexts ifs];
end

figure('Name',titl)
title(titl)

hold on

ylim([min(raster_stack(:,1)) max(raster_stack(:,1))])

plot([0 0],ylim,'-','LineWidth',2,'Color','b')
if chron==0
    plot(raster_stack(:,3),raster_stack(:,1),'-','LineWidth',2,'Color','b')
end

scatter((raster_stack(:,2)), raster_stack(:,1), 3, 'k', "filled");

xlim([pmin pmax])

ylim([min(raster_stack(:,1)) max(raster_stack(:,1))])
xticks([pmin pmin/2 0 pmax/2 pmax])
yticks([])
set(gca,'Ydir','reverse')

xlabel('Latency from inhalation (ms)')
box on
yt(txtsize)
set(gca,'LineWidth',2)

set(gcf,'Position', [ 0 0 700 1100])
set(gca,'Position', [ 0.04 0.06 0.93 0.9])
pause(2)
print(fullfile(out_dir, [titl ' ' num2str(pmax) ' ms raster.png']), '-dpng','-r600')

%% reference outputs for checking the Python port

out = struct();
out.titl = titl;
out.region = region;
out.cl = cl;
out.cll = cll;
out.params = struct('pmax',pmax,'pmin',pmin,'bwidth',bwidth, ...
    'fredge_edges',fredge_edges,'vio',vio,'chron',chron,'inl',inl, ...
    'Fs',Fs,'maxax',maxax);
out.locs = locs;
out.durs = durs;
out.spikes = spikes;
out.rpv = rpv;
out.over = over;
out.passes_gate = passes_gate;
out.snff_raw = snff_raw;
out.stime_axis = stime_axis;
out.sfreq_axis = sfreq_axis;
out.fredges = fredges;
out.xax = xax;
out.limn_raw = limn_raw;
out.limn_smooth = limn_smooth;
out.limn = limn;
out.sps_lim = sps_lim;
out.snff = snff;
out.breath_ind = breath_ind;
out.ord = ord;
out.raster_stack = raster_stack;

save(fullfile(out_dir, [titl '_reference.mat']), '-struct', 'out', '-v7')
end
