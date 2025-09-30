clc;
clear;

load("opt_errdata.mat");

T = 50000;
x = 1:T+1;
origin = 1;
idx_data = 1:2500:50001;
idx_marker = 2500:2500:50000;
colors5 = [176,7,14;255,140,0;0,100,0;0,0,139;139,0,139;] / 256;
figure('Units','inches','Position',[1 1 16 9]); 

h1 = semilogy(1, origin,'d-', 'Color', colors5(1,:),'LineWidth', 2, 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15);hold on;grid on;
semilogy(x(idx_data), opt_errdata_DSGD(idx_data),'-', 'Color', colors5(1,:),'LineWidth', 2); 
semilogy(x(idx_marker), opt_errdata_DSGD(idx_marker), 'd', 'Color', colors5(1,:), 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15, 'LineWidth', 2);

h2 = semilogy(1, origin,'o-', 'Color', colors5(2,:),'LineWidth', 2, 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15);
semilogy(x(idx_data), opt_errdata_CHOCO_Top(idx_data),'-', 'Color', colors5(2,:),'LineWidth', 2); 
semilogy(x(idx_marker), opt_errdata_CHOCO_Top(idx_marker), 'o', 'Color', colors5(2,:), 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15, 'LineWidth', 2);

h3 = semilogy(1, origin,'<-', 'Color', colors5(3,:),'LineWidth', 2, 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15);
semilogy(x(idx_data), opt_errdata_MoTEF_Top(idx_data),'-', 'Color', colors5(3,:),'LineWidth', 2); 
semilogy(x(idx_marker), opt_errdata_MoTEF_Top(idx_marker), '<', 'Color', colors5(3,:), 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15, 'LineWidth', 2);

h4 = semilogy(1, origin,'s-', 'Color', colors5(4,:),'LineWidth', 2, 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15);
semilogy(x(idx_data), opt_errdata_LBGD_Sign(idx_data),'-', 'Color', colors5(4,:),'LineWidth', 2); 
semilogy(x(idx_marker), opt_errdata_LBGD_Sign(idx_marker), 's', 'Color', colors5(4,:), 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15, 'LineWidth', 2);

h5 = semilogy(1, origin,'>-', 'Color', colors5(5,:),'LineWidth', 4, 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15);
semilogy(x(idx_data), opt_errdata_LBGD_HarMo(idx_data),'-', 'Color', colors5(5,:),'LineWidth', 4); 
semilogy(x(idx_marker), opt_errdata_LBGD_HarMo(idx_marker), '>', 'Color', colors5(5,:), 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15, 'LineWidth', 4);

xlim([0 4e4]);
ylim([0.3e-4 0.5e1]); 

xlabel('Iterations','FontName','Times New Roman', 'FontSize', 36);
ylabel(['$\frac{1}{n}\sum_{i=1}^n \left\| x_{i,t} - x^\star \right\|^{2}$'],'Interpreter','latex','FontName','Times New Roman', 'FontSize', 36);
legend([h1,h2,h3,h4,h5],{'DSGD', 'CHOCO\newline Top-\alpha', 'MoTEF\newline Top-\alpha', 'LBGD\newline Sign', '\bf LBGD\newline HarMo'},'FontName','Times New Roman', 'FontSize', 36, 'Orientation','horizontal');
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 36;   
grid on;
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
