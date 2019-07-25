inputFile = 'results/BK9NVIDIAV100AWS.dat';
outputFile = 'BK9NVIDIAV100AWS.pdf'
titleString = 'BK9:NVIDIA V100 SXM2:OCCA:CUDA';  
data = load(inputFile);

minN = min(data(:,1));
maxN = max(data(:,1));

minMode = min(data(:,end));
maxMode = max(data(:,end));

measN = size(data,2);

clf

forms{1} = 'r--';
forms{2} = 'g-h';
forms{3} = 'b-';
forms{4} = 'm-*';
forms{5} = 'c-+';
forms{6} = 'k-s';
forms{7} = 'g-o';
forms{8} = 'b-d';

scols{1} = 'k';
scols{2} = 'r';
scols{3} = 'g';
scols{4} = 'b';
scols{5} = 'k';
scols{6} = 'g';
scols{7} = 'b';
scols{8} = 'r';

hold on

idBW = 9;
idGB = 6;
idVNODES = 4;
for N=minN:maxN
  ids = find(data(:,1)==N);
  dataN = data(ids,:);

  Nsamps = 1e9;
  for mode=minMode:maxMode
    idsM = find(dataN(:,end)==mode);
    Nsamps = min(Nsamps, length(idsM));
  end

  dataNMAX = zeros(Nsamps, measN);
  for mode=minMode:maxMode
    idsM = find(dataN(:,end)==mode);
    dataNM = dataN(idsM,:);
    for n=1:Nsamps
      if(dataNM(n,idBW)>dataNMAX(n,idBW))
	dataNMAX(n,:) = dataNM(n,:);
      end
    end
  end

  
  ha = plot(dataNMAX(:,idVNODES), dataNMAX(:,idBW), forms{N});
  set(ha, 'MarkerFaceColor', scols{N})
  set(ha, 'MarkerEdgeColor', scols{N})
  dataNMAXsave{N} = dataNMAX;
  

  legs{N-1} = sprintf('N_v=%d, N_p=%d', N, N-1);
  
end

for N=minN:maxN

%  dataNMAXsave{N} = dataNMAX;
  r = 0.8;
  [maxBW] = max(dataNMAXsave{N}(:,idBW));
  ids = find(dataNMAXsave{N}(:,idBW)>r*maxBW);
  rid = ids(1);
  rVN = dataNMAXsave{N}(rid,idVNODES);
  rBW = dataNMAXsave{N}(rid,idBW);
  ha = plot([rVN,rVN], [0;rBW], forms{N});
  set(ha, 'MarkerFaceColor', scols{N})
  set(ha, 'MarkerEdgeColor', scols{N})
  
end
  
hold off
legend(legs, 'location', 'southeast')

xlabel('#Velocity Nodes', 'FontSize', 16)
ylabel('Bandwidth (GB/s)', 'FontSize', 16)
  title(titleString, 'FontSize', 16)
grid on
box on
  grid minor
  
  axis([0 1e6 0 800])
  print('-bestfit', '-dpdf','-painters', outputFile)
  
