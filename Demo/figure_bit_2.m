clc;
clear;

load("opt_errdata_bit_2.mat");

bit_DSGD = 32*2*2*9*2000;
bit_CHOCO_Top = 64*2*2*9*200;
bit_MoTEF_Top = 64*2*2*9*200;
bit_LBGD_Sign = 1*2*2*9*(2000+32);
bit_LBGD_HarMo = 16*2*2*9*1;
colors5 = [176,7,14;255,140,0;0,100,0;0,0,139;139,0,139;] / 256;
figure('Units','inches','Position',[1 1 16 9]);

h1 = loglog(0, 1,'d-', 'Color', colors5(1,:),'LineWidth', 2, 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15);hold on;grid on;
loglog(opt_errdata_bit_DSGD_1, opt_errdata_result_DSGD_1 ,'-', 'Color', colors5(1,:),'LineWidth', 2, 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15); 
loglog(opt_errdata_bit_DSGD_2,  opt_errdata_result_DSGD_2, 'd', 'Color', colors5(1,:), 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15, 'LineWidth', 1);

h2 = loglog(0, 1,'o-', 'Color', colors5(2,:),'LineWidth', 2, 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_CHOCO_Top_1, opt_errdata_result_CHOCO_Top_1,'-', 'Color', colors5(2,:),'LineWidth', 2, 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_CHOCO_Top_2, opt_errdata_result_CHOCO_Top_2, 'o', 'Color', colors5(2,:), 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15, 'LineWidth', 1);

h3 = loglog(0, 1,'<-', 'Color', colors5(3,:),'LineWidth', 2, 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_MoTEF_Top_1, opt_errdata_result_MoTEF_Top_1,'-', 'Color', colors5(3,:),'LineWidth', 2, 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_MoTEF_Top_2, opt_errdata_result_MoTEF_Top_2, '<', 'Color', colors5(3,:), 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15, 'LineWidth', 1);

h4 = loglog(0, 1,'s-', 'Color', colors5(4,:),'LineWidth', 2, 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_LBGD_Sign_1, opt_errdata_result_LBGD_Sign_1,'-', 'Color', colors5(4,:),'LineWidth', 2, 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_LBGD_Sign_2, opt_errdata_result_LBGD_Sign_2, 's', 'Color', colors5(4,:), 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15, 'LineWidth', 1);

h5 = loglog(0, 1,'>-', 'Color', colors5(5,:),'LineWidth', 4, 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_LBGD_HarMo_1, opt_errdata_result_LBGD_HarMo_1,'-', 'Color', colors5(5,:),'LineWidth', 4, 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15);
loglog(opt_errdata_bit_LBGD_HarMo_2, opt_errdata_result_LBGD_HarMo_2, '>', 'Color', colors5(5,:), 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15, 'LineWidth', 1);

xlim([10e2 10e11]);
ylim([0.0078 1.5]);
xlabel('Comunicated bits','FontName','Times New Roman', 'FontSize', 36);
ylabel('$f(\bar{x}_t) - f(x^\star)$', 'Interpreter','latex','FontName','Times New Roman', 'FontSize', 36);
legend([h1,h2,h3,h4,h5],{'DSGD', 'CHOCO\newline Top-\alpha', 'MoTEF\newline Top-\alpha', 'LBGD\newline Sign', '\bf  LBGD\newline HarMo'},'FontName','Times New Roman', 'FontSize', 36, 'Orientation','horizontal');
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 36;
grid on;
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

