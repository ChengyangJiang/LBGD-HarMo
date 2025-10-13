clc;
clear;

load("opt_errdata_convergence_2.mat");

idx(1:20)= 1:1:20;
idx(21:119) = 40:20:2000;
idx(120:209) = 2200:200:20000;
idx(210:299) = 22000:2000:200000;
idx(300:359) = 210000:10000:800000;
idx_marker_1 = 50000:50000:800000;
idx_marker_loss_1 = [224,249,274,299,304,309,314,319,324,329,334,339,344,349,354,359];
idx_marker_2 = 25000:50000:800000;
idx_marker_loss_2 = [211,236,261,286,301,306,311,316,321,326,331,336,341,346,351,356];
idx_data = [1  122 211	224	236	249	261	274	286	299	301	304	306	309	311	314	316	319	321	324	326	329	331	334	336	339	341	344	346	349	351	354	356	359];
colors5 = [176,7,14;255,140,0;0,100,0;0,0,139;139,0,139;] / 256;
figure('Units','inches','Position',[1 1 16 9]);

h1 = semilogy(1, opt_errdata_result_DSGD(1),'d-', 'Color', colors5(1,:),'LineWidth', 2, 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15);hold on;grid on;
semilogy(idx(idx_data), opt_errdata_result_DSGD(idx_data),'-', 'Color', colors5(1,:),'LineWidth', 2, 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15); 
semilogy(idx_marker_1, opt_errdata_result_DSGD(idx_marker_loss_1), 'd', 'Color', colors5(1,:), 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15, 'LineWidth', 1);
semilogy(idx_marker_2, opt_errdata_result_DSGD(idx_marker_loss_2), 'd', 'Color', colors5(1,:), 'MarkerFaceColor', colors5(1,:), 'MarkerSize', 15, 'LineWidth', 1);

h2 = semilogy(1, opt_errdata_result_CHOCO_Top(1),'o-', 'Color', colors5(2,:),'LineWidth', 2, 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15);
semilogy(idx(idx_data), opt_errdata_result_CHOCO_Top(idx_data),'-', 'Color', colors5(2,:),'LineWidth', 2, 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15);
semilogy(idx_marker_1, opt_errdata_result_CHOCO_Top(idx_marker_loss_1), 'o', 'Color', colors5(2,:), 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15, 'LineWidth', 1);
semilogy(idx_marker_2, opt_errdata_result_CHOCO_Top(idx_marker_loss_2), 'o', 'Color', colors5(2,:), 'MarkerFaceColor', colors5(2,:), 'MarkerSize', 15, 'LineWidth', 1);

h3 = semilogy(1, opt_errdata_result_MoTEF_Top(1),'<-', 'Color', colors5(3,:),'LineWidth', 2, 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15);
semilogy(idx(idx_data), opt_errdata_result_MoTEF_Top(idx_data),'-', 'Color', colors5(3,:),'LineWidth', 2, 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15);
semilogy(idx_marker_1, opt_errdata_result_MoTEF_Top(idx_marker_loss_1), '<', 'Color', colors5(3,:), 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15, 'LineWidth', 1);
semilogy(idx_marker_2, opt_errdata_result_MoTEF_Top(idx_marker_loss_2), '<', 'Color', colors5(3,:), 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15, 'LineWidth', 1);
semilogy(2600, opt_errdata_result_MoTEF_Top(122), '<', 'Color', colors5(3,:), 'MarkerFaceColor', colors5(3,:), 'MarkerSize', 15, 'LineWidth', 1);

h4 = semilogy(1, opt_errdata_result_LBGD_Sign(1),'s-', 'Color', colors5(4,:),'LineWidth', 2, 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15);
semilogy(idx(idx_data), opt_errdata_result_LBGD_Sign(idx_data),'-', 'Color', colors5(4,:),'LineWidth', 2, 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15);
semilogy(idx_marker_1, opt_errdata_result_LBGD_Sign(idx_marker_loss_1), 's', 'Color', colors5(4,:), 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15, 'LineWidth', 1);
semilogy(idx_marker_2, opt_errdata_result_LBGD_Sign(idx_marker_loss_2), 's', 'Color', colors5(4,:), 'MarkerFaceColor', colors5(4,:), 'MarkerSize', 15, 'LineWidth', 1);

h5 = semilogy(1, opt_errdata_result_LBGD_HarMo(1),'>-', 'Color', colors5(5,:),'LineWidth', 4, 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15);
semilogy(idx(idx_data), opt_errdata_result_LBGD_HarMo(idx_data),'-', 'Color', colors5(5,:),'LineWidth', 4, 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15);
semilogy(idx_marker_1, opt_errdata_result_LBGD_HarMo(idx_marker_loss_1), '>', 'Color', colors5(5,:), 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15, 'LineWidth', 1);
semilogy(idx_marker_2, opt_errdata_result_LBGD_HarMo(idx_marker_loss_2), '>', 'Color', colors5(5,:), 'MarkerFaceColor', colors5(5,:), 'MarkerSize', 15, 'LineWidth', 1);

xlim([0 4e5]);
ylim([0.001 0.7]);
xlabel('Iterations','FontName','Times New Roman', 'FontSize', 40);
ylabel('$f(\bar{x}_t) - f(x^\star)$', 'Interpreter','latex','FontName','Times New Roman', 'FontSize', 40);
legend([h1,h2,h3,h4,h5],{'DSGD', 'CHOCO\newline Top-\alpha', 'MoTEF\newline Top-\alpha', 'LBGD\newline Sign', '\bf LBGD\newline HarMo'},'FontName','Times New Roman', 'FontSize', 36, 'Orientation','horizontal');
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 40;     
grid on;
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
