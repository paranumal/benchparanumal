

if(0)
bandwidthTestNVIDIATitanV
titleString = 'Bandwidth Test:NVIDIA Titan V:OCCA:CUDA'
pdfName = 'bandwidthTestNVIDIATitanV.pdf'
end

if(1)
bandwidthTestNVIDIAV100AWS
titleString = 'Bandwidth Test:NVIDIA V100 SXM2:OCCA:CUDA'
pdfName = 'bandwidthTestNVIDIAV100.pdf'
end

if(0)
bandwidthTestAMDRadeonVII
titleString = 'Bandwidth Test:AMD Radeon VII:OCCA:HIP'
pdfName = 'bandwidthTestAMDRadeonVII.pdf'
end
  
knl00 = memcpyKNL00;
knl01 = memcpyKNL01;
knl02 = memcpyKNL02;
knlMC = memcpyBW;

haMC = plot(knlMC(:,1), knlMC(:,3));

hold on 
ha00 = plot(knl00(:,1), knl00(:,3));
ha01 = plot(knl01(:,1), knl01(:,3));
ha02 = plot(knl02(:,1), knl02(:,3));
hold off
  
axis([0 2e8 0 900])

r = 0.8;

subids = find(knlMC(:,1)>10e6);

knlMC = knlMC(subids,:);
knl00 = knl00(subids,:);
knl01 = knl01(subids,:);
knl02 = knl02(subids,:);

[maxKnlMC] = max(knlMC(:,3));
ids = find(knlMC(:,3)>r*maxKnlMC);
knlMCn08 = knlMC(ids(1),1);
knlMCbw08 = knlMC(ids(1),3);

[maxKnl00] = max(knl00(:,3));
ids = find(knl00(:,3)>r*maxKnl00);
knl00n08 = knl00(ids(1),1);
knl00bw08 = knl00(ids(1),3);

[maxKnl01] = max(knl01(:,3));
ids = find(knl01(:,3)>r*maxKnl01);
knl01n08 = knl01(ids(1),1);
knl01bw08 = knl01(ids(1),3);

[maxKnl02] = max(knl02(:,3));
ids = find(knl02(:,3)>r*maxKnl02);
knl02n08 = knl02(ids(1),1);
knl02bw08 = knl02(ids(1),3);


hold on
plot([knlMCn08,knlMCn08], [0,knlMCbw08], '--', 'color', get(haMC, 'color'))
plot([knl00n08,knl00n08], [0,knl00bw08], '--', 'color', get(ha00, 'color'))
plot([knl01n08,knl01n08], [0,knl01bw08], '--', 'color', get(ha01, 'color'))
plot([knl02n08,knl02n08], [0,knl02bw08], '--', 'color', get(ha02, 'color'))
hold off

ha = legend('memcpy', 'KNL 00: write',  'KNL 01: read+write', 'KNL 02: read', 'location', 'southeast')
set(ha, 'FontSize', 16)
box
grid minor

xlabel('# Bytes moved', 'FontSize', 16);
ylabel('Bandwidth (GB/s)', 'FontSize', 16);
title(titleString, 'FontSize', 16)

print('-dpdf', '-bestfit', '-painters', pdfName)



