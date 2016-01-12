mooc_load;

tree = fitrtree(x1_train, y_train, 'CrossVal', 'on', 'HoldOut', 0.05);

yp_train = predict(tree, x1_train);
ey_train = y_train - yp_train;
rms_train = sqrt(mean(ey_train .* ey_train));
yp_val = predict(tree, x1_val);
ey_val = y_val - yp_val;
rms_val = sqrt(mean(ey_val .* ey_val));
