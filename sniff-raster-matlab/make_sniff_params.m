clear
src='E:/cbm-odor/preprocessed/';%top-level data folder; searched recursively for sniff files
events_dir='E:/cbm-odor/events/';%<mouse>/<session>/events.csv; last timestamp_ms is the end of the session
out_dir='E:/cbm-odor/sniff/';%outputs are saved to <mouse>/<session>/ in here
inv_file=fullfile(out_dir,'sniff_inventory.csv');%one row per session: saved or why it was rejected
redo=false;%true reprocesses sessions that already have sniff_params.mat
Fs_raw=30e3;%acquisition board sampling rate of sniff.npy
Fs=1e3;%sniff is downsampled to this so that all times are in ms
sniffpile=dir(fullfile(src,'**','*sniff.npy'));
scanwindow=2e3;
smoo=25;%smoothing span (ms)
min_peak_dist=50;%ms; only the tallest peak within this distance counts, so small bumps next to a real sniff are ignored
excluded_mice={'9004'};%known poor sniff signal; always rejected, but still saved for inspection
%%
for sp=1:length(sniffpile)
    fold=sniffpile(sp).folder;
    names=strsplit(fold,{'/','\'});
    mouse=names{end-1};
    session=names{end};
    save_dir=fullfile(out_dir,mouse,session);

    if ~redo && exist(fullfile(save_dir,'sniff_params.mat'),'file')
        continue %already processed
    end

    %quality numbers for the inventory; stay NaN if the session stops early
    session_end=nan; n_good=nan; n_bad=nan; gate_ratio=nan; mean_f=nan;

    ev_file=fullfile(events_dir,mouse,session,'events.csv');
    if ~exist(ev_file,'file')
        warning('No events.csv for %s, skipping',fold)
        log_session(inv_file,mouse,session,"no events.csv",session_end,n_good,n_bad,gate_ratio,mean_f)
        continue
    end
    ev_opts=detectImportOptions(ev_file);
    ev_opts.SelectedVariableNames={'timestamp_ms'};
    ev_ts=readmatrix(ev_file,ev_opts);
    session_end=max(ev_ts);%ms; the sniff sensor is often unplugged after this

    sniff = double(readNPY(fullfile(fold,sniffpile(sp).name)));

    if ~isempty(sniff)
        sniff = resample(sniff,Fs,Fs_raw);
        sniff = sniff(1:min(end,floor(session_end)));%cut off everything after the session ends
        sniff = smooth(sniff,smoo,'sgolay');

        sniff=sniff(:); %turn from row to column vector (easier for me to think this way)

        scanner=1:scanwindow:length(sniff);%break the data into scanwindow ms chunks
        %declare the variables for the sniff parameters
        i_locs=[];%%inhalation start times
        e_locs=[];%end of inhalation times

        for scan=2:(length(scanner)-1)
            %grabs the current window; notice that it grabs three of the
            %scanner points: scan-1, scan , and scan+1. this is how it does
            %overlapping windows
            znindow=scanner(scan-1):scanner(scan+1);
            whiff=sniff(znindow);%take as a sample from the sniff signal
            zniff=zscore(whiff);%zscoring scales the signal so that the peak detection is reliable

            [~,in_locs] = findpeaks(zniff,'MinPeakDistance',min_peak_dist,'MinPeakProminence',0.5); %find inhalation points
            [~,ex_locs] = findpeaks(-zniff,'MinPeakDistance',min_peak_dist,'MinPeakProminence',0.5); %find exhalation points

            i_locs=[i_locs; in_locs+scanner(scan-1)];
            e_locs=[e_locs; ex_locs+scanner(scan-1)];

            % a piece of code where you can watch examples to
            % quality control

            % figure
            % plot(whiff,'k-')
            % hold on
            % scatter(in_locs, whiff(in_locs),24,'r','filled')
            %
            % close
        end
        %gets rid of duplicates due to the overlapping window
        in_amps=find(unique(i_locs));
        ex_amps=find(unique(e_locs));
        i_locs=unique(i_locs);
        e_locs=unique(e_locs);

        bad_locs=[];%declare variable for rejected peaks
        fsniff=(Fs./(diff(i_locs)));%instantaneous frequency
        mean_f=mean(fsniff);
        %sessions with a mean sniff rate >15 Hz are rejected below, but are still cleaned
        %up the same way so their detections can be inspected
        cuts=find(fsniff>17 | fsniff<0.5);%find unrealistic sniffs to reject

        %loop through suspect sniffs
        for bl=1:length(cuts)

            glind=cuts(bl)+[-2:2];%window of +/- 2 sniffs around the suspect inhalation peak

            glind=glind(glind>0 & glind<size(i_locs,1));%just so it doesn't go off the edge

            bad_locs=[bad_locs; i_locs(glind)];% add all the inhalation times in the suspect window

            interv=i_locs(glind(1)):i_locs(glind(end));%define a window between the suspects to eliminate the corresponding entries in the other arrays

            eglint=intersect(e_locs,interv);%find the time of all the suspect end times
            [~,eglind]=ismember(eglint,e_locs);%find the index of all the suspect end times
            i_locs(glind)=nan;
            e_locs(eglind)=nan;

            in_amps(glind)=nan;
            ex_amps(eglind)=nan;
        end
        %this code eliminates times in the signal where it clips or
        %flatlines
        clips=find(diff(sniff,6)==0);%if six consecutive samples are the same value -> suspect
        %the first and last (smoo-1)/2 samples are a single polynomial fit from the smoothing,
        %so their 6th difference is ~0 by construction and only rounding decides if it is exactly 0
        edge_n=(smoo-1)/2;
        clips=clips(clips>edge_n & clips+6<=length(sniff)-edge_n);
        %logic for this loop is the same as the cut finding loop
        for bl=1:length(clips)

            satind=clips(bl)+[-3*(smoo):(3*(smoo))];

            iclipd=intersect(satind,i_locs);
            eclipd=intersect(satind,e_locs);

            if ~isempty(iclipd)
                iclipd=iclipd(1);
                clind=find(i_locs==iclipd);
                clint=clind+[-6:6];

                clint=clint(clint>0 & clint<size(i_locs,1));
                bad_locs=[bad_locs; i_locs(clint)];

                i_locs(clint)=nan;
                in_amps(clint)=nan;
            end

            if ~isempty(eclipd)
                eclipd=eclipd(1);
                clind=find(e_locs==eclipd);
                clint=clind+[-6:6];
                clint=clint(clint>0 &clint<size(e_locs,1));

                e_locs(clint)=nan;
                ex_amps(clint)=nan;
            end
        end
        %instantaneous frequency of the remaining good sniffs.
        %also, when you take the diff and it hits a nan it makes the before
        %and after into nans. Helpful for ensuring everything suspect is
        %cut out
        ffsniff=(Fs./(diff(i_locs)));

        xfsniff=(Fs./diff(e_locs));

        bad_locs=[bad_locs; i_locs(isnan(ffsniff))];
        i_locs=i_locs(~isnan(ffsniff));
        in_amps=in_amps(~isnan(ffsniff));
        e_locs=e_locs(~isnan(xfsniff));
        ex_amps=ex_amps(~isnan(xfsniff));

        n_good=length(i_locs);
        n_bad=numel(unique(bad_locs(~isnan(bad_locs))));
        gate_ratio=n_bad/n_good;%fraction the check below uses
        if ismember(mouse,excluded_mice)
            status="rejected: excluded mouse (poor sniff signal)";
        elseif mean_f>15
            status="rejected: mean sniff rate >15 Hz";
        elseif ~(~isempty(i_locs) && n_bad<n_good/10)%each rejected inhalation counted once
            status="rejected: too many bad sniffs";
        else
            %this bit of code makes sure that every i_loc is matched with 1
            %e_loc
            [~,~,ebn]=histcounts(e_locs,i_locs);

            e_locs=e_locs(diff(ebn)>0);
            ex_amps=ex_amps(diff(ebn)>0);

            [~,~,ebn]=histcounts(e_locs,i_locs);

            badset=setdiff(1:length(i_locs),ebn);
            bad_locs=[bad_locs; i_locs(badset)];

            i_locs=i_locs(ebn(ebn>0));
            in_amps=in_amps(ebn(ebn>0));

            bad_locs=bad_locs(~isnan(bad_locs));
            bad_locs=unique(bad_locs);

            if e_locs(1)<i_locs(1)
                e_locs=e_locs(2:end);ex_amps=ex_amps(2:end);
            end
            padd=zeros(length(bad_locs),1);

            %at the suspect inhalation times, set the other parameters to
            %zero
            i_locs=[i_locs; -bad_locs];
            e_locs=[e_locs; padd];
            in_amps=[in_amps; padd];
            ex_amps=[ex_amps; padd];

            [~,bmx]=sort(abs(i_locs));

            i_locs=i_locs(bmx);
            e_locs=e_locs(bmx);
            in_amps=in_amps(bmx);
            ex_amps=ex_amps(bmx);

            i_locs=abs(i_locs);
            %make a matrix with all the parameters as columns
            sniff_params=[i_locs in_amps e_locs ex_amps];

            status="saved";
        end

        %save every session, so rejected ones can be looked at too. rejected sessions get
        %rejected_sniffs.mat instead of sniff_params.mat, so they are never picked up as good
        if ~exist(save_dir,'dir')
            mkdir(save_dir)
        end
        save(fullfile(save_dir,'sniff_signal.mat'),'sniff');%save the resampled sniff signal (sampling rate 1kHz)
        if status=="saved"
            save(fullfile(save_dir,'sniff_params.mat'),'sniff_params')%save sniff parameters
            delete_if_exists(fullfile(save_dir,'rejected_sniffs.mat'))
        else
            %good inhalations, exhalations and rejected inhalations at the point the session was rejected
            bad_locs=unique(bad_locs(~isnan(bad_locs)));
            save(fullfile(save_dir,'rejected_sniffs.mat'),'i_locs','e_locs','bad_locs')
            delete_if_exists(fullfile(save_dir,'sniff_params.mat'))
        end
    else
        status="empty sniff.npy";
    end
    log_session(inv_file,mouse,session,status,session_end,n_good,n_bad,gate_ratio,mean_f)
end

function log_session(inv_file,mouse,session,status,session_end,n_good,n_bad,gate_ratio,mean_f)
%add or replace this session's row in the inventory csv
row=table(string(mouse),string(session),status,session_end/6e4,n_good,n_bad, ...
    100*n_bad/(n_good+n_bad),100*gate_ratio,mean_f,string(datetime('now','Format','yyyy-MM-dd HH:mm')), ...
    'VariableNames',{'mouse','session','status','session_end_min','n_good_inhalations', ...
    'n_bad_inhalations','pct_bad','gate_pct','mean_sniff_hz','processed'});
if exist(inv_file,'file')
    opts=detectImportOptions(inv_file);
    opts=setvartype(opts,{'mouse','session','status','processed'},'string');
    inv=readtable(inv_file,opts);
    inv=inv(~(inv.mouse==row.mouse & inv.session==row.session),:);
    inv=[inv; row];
else
    if ~exist(fileparts(inv_file),'dir')
        mkdir(fileparts(inv_file))
    end
    inv=row;
end
inv=sortrows(inv,{'mouse','session'});
writetable(inv,inv_file)
end


function delete_if_exists(f)
if exist(f,'file')
    delete(f)
end
end
