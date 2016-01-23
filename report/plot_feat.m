clf;
hold on; grid on;

h = bar([87.64, 87.80; 79.33, 79.22]);
colormap jet;
ylim([75, 90]);

set(gca, 'XTick', [1, 2]);
set(gca, 'XTickLabel', {'feat1', 'featTA'});
ylabel('Accuracy(\%)');
legend('Development', 'Validation');

neat;
export_fig('fig/feat', '-pdf');
