clc;
clear;

load("opt_errdata_bit_1.mat");

T = 50000;
x = 1:T;
origin = 2;
idx_data = 1:1:50000;
idx_marker = 1:2000:50000;
colors5 = [176,7,14;255,140,0;0,100,0;0,0,139;139,0,139;] / 256;

bit_DSGD = 32*2*2*25*8;
bit_CHOCO_Top = 64*2*2*25*1;
bit_MoTEF_Top = 64*2*2*25*1;
bit_LBGD_Sign = 1*2*2*25*(8+32);
bit_LBGD = 8*2*2*25*1;

bit_DSGD_all = 0:bit_DSGD:bit_DSGD*(T);
bit_CHOCO_Top_all = 0:bit_CHOCO_Top:bit_CHOCO_Top*(T-1);
bit_MoTEF_Top_all = 0:bit_MoTEF_Top:bit_MoTEF_Top*(T-1);
bit_LBGD_Sign_all = 0:bit_LBGD_Sign:bit_LBGD_Sign*(T);
bit_LBGD_all = 0:bit_LBGD:bit_LBGD*(T);
opt_errdata_DSGD(50001)= opt_errdata_DSGD(50000);
opt_errdata_LBGD_Sign(50001)= opt_errdata_LBGD_Sign(50000);
opt_errdata_LBGD_HarMo(50001) = opt_errdata_LBGD_HarMo(50000);

opt_errdata_DSGD(1) = 1;
opt_errdata_CHOCO_Top(1)=1;
opt_errdata_MoTEF_Top(1)=1;
opt_errdata_LBGD_Sign(1)=1;
opt_errdata_LBGD_HarMo(1)=1;

figure('Units','inches','Position',[1 1 16 9]); 
h1 = semilogy(100000, opt_errdata_DSGD(1),'d-', 'Color', colors5(1,:),'LineWidth', 2, 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15);hold on;grid on;
semilogy(bit_DSGD_all(1:98:50001), opt_errdata_DSGD(1:98:50001),'-', 'Color', colors5(1,:),'LineWidth', 2); 
semilogy(bit_DSGD_all(1:98:50001), opt_errdata_DSGD(1:98:50001), 'd', 'Color', colors5(1,:), 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15, 'LineWidth', 2);
semilogy(bit_DSGD_all(782), opt_errdata_DSGD(782), 'd', 'Color', colors5(1,:), 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15, 'LineWidth', 2);

h2 = semilogy(100000, opt_errdata_CHOCO_Top(1),'o-', 'Color', colors5(2,:),'LineWidth', 2, 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15);
semilogy(bit_CHOCO_Top_all(1:390:50000), opt_errdata_CHOCO_Top(1:390:50000),'-', 'Color', colors5(2,:),'LineWidth', 2); 
semilogy(bit_CHOCO_Top_all(1:390:50000), opt_errdata_CHOCO_Top(1:390:50000), 'o', 'Color', colors5(2,:), 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15, 'LineWidth', 2);

h3 = semilogy(100000, opt_errdata_MoTEF_Top(1),'<-', 'Color', colors5(3,:),'LineWidth', 2, 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15);
semilogy(bit_MoTEF_Top_all(1:390:50000), opt_errdata_MoTEF_Top(1:390:50000),'-', 'Color', colors5(3,:),'LineWidth', 2); 
semilogy(bit_MoTEF_Top_all(1:390:50000), opt_errdata_MoTEF_Top(1:390:50000), '<', 'Color', colors5(3,:), 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15, 'LineWidth', 2);

h4 = semilogy(100000,opt_errdata_LBGD_Sign(1),'s-', 'Color', colors5(4,:),'LineWidth', 2, 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15);
semilogy(bit_LBGD_Sign_all(1:694:50001), opt_errdata_LBGD_Sign(1:694:50001),'-', 'Color', colors5(4,:),'LineWidth', 2); 
semilogy(bit_LBGD_Sign_all(695:694:50000), opt_errdata_LBGD_Sign(695:694:50000), 's', 'Color', colors5(4,:), 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15, 'LineWidth', 2);

h5 = semilogy(100000, opt_errdata_LBGD_HarMo(1),'>-', 'Color', colors5(5,:),'LineWidth', 4, 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15);
semilogy(bit_LBGD_all(1:3125:50001), opt_errdata_LBGD_HarMo(1:3125:50001),'-', 'Color', colors5(5,:),'LineWidth', 4); 
semilogy(bit_LBGD_all(3125:3125:50000), opt_errdata_LBGD_HarMo(3125:3125:50000), '>', 'Color', colors5(5,:), 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15, 'LineWidth', 4);

xlim([4e4 4e7]);
ylim([0.3e-4 0.5e1]); 

xlabel('Comunicated bits','FontName','Times New Roman', 'FontSize', 36);
ylabel('$\frac{1}{n}\sum_{i=1}^n \left\| x_{i,t} - x^\star \right\|^{2}$','Interpreter','latex','FontName','Times New Roman', 'FontSize', 36);
legend([h1,h2,h3,h4,h5],{'DSGD', 'CHOCO\newline Top-\alpha', 'MoTEF\newline Top-\alpha', 'LBGD\newline Sign', '\bf LBGD\newline HarMo'},'FontName','Times New Roman', 'FontSize', 36, 'Orientation','horizontal');
ax = gca; 
ax.FontName = 'Times New Roman';  
ax.FontSize = 36; 
grid on;
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
