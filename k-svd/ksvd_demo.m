load('trainimgs.mat');
load('trainlabs.mat');
load('testimgs.mat');
load('testlabs.mat');
train_num = 1000;
test_num = 100;
X_train = trainimgs(:,1:train_num); %d*n
y_train = trainlabs(1:train_num);
X_test = testimgs(:, 1:test_num);
y_test = testlabs(1:test_num);
% use ksvd to create dictionary
[D, X] = k_svd(X_train, 10);


y_prediction = prediction(D, X, y_train, X_test);
count = 0;
for i = 1:100
    if y_prediction(i) == y_test(i)
        count = count + 1;
    end
end
accuracy = count/num_test;
fprintf('The accuracy of k-svd is %f\n', accuracy);