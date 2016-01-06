make_feat1;

m = cell2mat(cellfun(@(x, y) [x * ones(size(y)), y], mat2cell((1:index_max)', ones(index_max, 1), 1), c, 'UniformOutput', false));
session_duration_avg = cellfun(@mean, c);

figure(sum('main0')); clf;
hold on; grid on;
colormap jet;

scatter(m(:, 1), m(:, 2));
plot(1:index_max, session_duration_avg, 'LineWidth', 3);
scatter((1:index_max)', session_duration_avg, 50, truth.dropout(1:index_max), 'filled');

set(gca, 'yScale', 'log');
neat_large;

figure(sum('main1'));
colormap jet;

imagesc(log(x + 1));
colorbar;
