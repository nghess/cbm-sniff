clear
rng("shuffle")
sniff_dir='E:/cbm-odor/sniff/';%<mouse>/<session>/sniff_params.mat from make_sniff_params.m
ks_root='E:/cbm-odor/';%phy-curated spikes in <region>-ks/<mouse>/<session>/
fig_dir=fullfile(sniff_dir,'figs');%figures are saved to <mouse>/<session>/ in here
png_dir=fullfile(sniff_dir,'png');%.png copies of the figures, same layout
png_res=150;%dpi
only_sessions={};%e.g. {'9000/o2','9005/x3'} to run a few sessions; empty runs all

spike_havers=dir(fullfile(sniff_dir,'*','*','sniff_params.mat'));

%%
pmax=250;
pmin=-pmax;
wind=pmax;
bwidth=2;
fredge_edges=[2 13];
% set whether it's chronologically or sniff dur sorted
chron=0;%0=dur sort 1=chronological
%align to i_locs or e_locs
inl=1; %1 = inhalation ; 0 = exhalation
Fs=1e3;
%set time axis limits
maxax=1e4;

txtsize=20;

regions={'ob','hc'};
snf_stack=[];
for ss=1:length(spike_havers)
    this_fold=spike_havers(ss).folder;
    sp=strsplit(this_fold,{'/','\'});
    mouse_id=sp{end-1};
    date=sp{end};
    if ~isempty(only_sessions) && ~ismember([mouse_id '/' date],only_sessions)
        continue
    end
    out_dir=fullfile(fig_dir,mouse_id,date);
    out_png=fullfile(png_dir,mouse_id,date);
    if ~exist(out_dir,'dir')
        mkdir(out_dir)
    end
    if ~exist(out_png,'dir')
        mkdir(out_png)
    end

    load(fullfile(this_fold,'sniff_params.mat'),'sniff_params');

    i_locs=sniff_params(:,1);
    e_locs=sniff_params(:,3);

    % this code strips out the bad inhalation times
    b_locs=i_locs;
    b_locs(e_locs==0)=nan;

    durs=diff(b_locs);

    i_locs=i_locs(~isnan(durs));
    e_locs=e_locs(~isnan(durs));
    all_durs=durs(~isnan(durs));
    if inl==1
        all_locs=i_locs;
    elseif inl==0
        all_locs=e_locs;
    end

    %spikes are sorted separately per region. cluster ids are only unique within a region
    for rr=1:length(regions)
        region=regions{rr};
        ks_dir=fullfile(ks_root,[region '-ks'],mouse_id,date);

        if ~exist(fullfile(ks_dir,'cluster_info.tsv'),'file')
            fprintf('%s/%s %s: no cluster_info.tsv, skipping\n',mouse_id,date,region)
            continue
        end

        spike_times=readNPY(fullfile(ks_dir,'spike_times.npy'));
        clusters=readNPY(fullfile(ks_dir,'spike_clusters.npy'));
        %every unit that phy curation did not label noise (blank labels are kept)
        info=readtable(fullfile(ks_dir,'cluster_info.tsv'),'FileType','text','Delimiter','\t','TextType','string');
        clinclude=flipud(sort(info.cluster_id(info.group~="noise")));

        durs=all_durs(all_locs<spike_times(end)/30);
        locs=all_locs(all_locs<spike_times(end)/30);

        for  cll=1:length(clinclude)
            cl=clinclude(cll);

            titl=[mouse_id ' ' date ' ' region ' cl ' num2str((cl))];
            snf_file=fullfile(out_dir,[region ' cl ' num2str(cl) ' SnF.fig']);
            raster_file=fullfile(out_dir,[region ' cl ' num2str(cl) ' ' num2str(pmax) ' ms raster.fig']);
            snf_png=fullfile(out_png,[region ' cl ' num2str(cl) ' SnF.png']);
            raster_png=fullfile(out_png,[region ' cl ' num2str(cl) ' ' num2str(pmax) ' ms raster.png']);
            if isempty(locs) || all(cellfun(@(f) exist(f,'file')>0,{snf_file,raster_file,snf_png,raster_png}))
                continue
            end

            spikes=(double(spike_times(clusters==cl ))./30);

            try
                %% sniff field plot
                [snff,stime_axis,sfreq_axis,~,fredges,xax]=sniff_fielder16(spikes,locs,durs,pmax,bwidth,fredge_edges);

                snf_stack=cat(3,snf_stack,snff);


                limn=(Fs./fredges(1:(end)))./bwidth;

                limn=smooth(limn',11,'sgolay')';
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

                xlim([0 wind])
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


                savefig(gcf,snf_file)
                exportgraphics(gcf,snf_png,'Resolution',png_res)
                close all


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
                %%
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
                savefig(gcf,raster_file)
                exportgraphics(gcf,raster_png,'Resolution',png_res)
                close all
            catch err
                fprintf('%s: failed (%s), skipping\n',titl,err.message)
                close all
            end
        end
    end
end
