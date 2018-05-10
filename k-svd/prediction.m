function y_prediction =  prediction(D, X, label, data_test)
% predict the label
% INPUT:
% D: dictionary
% X: sparse coding
% data: test data
    classifier = TreeBagger(10, X', label);
    num_test = size(data_test, 2);
    y_prediction = zeros(1, num_test);
    for i = 1:num_test
        temp = data_test(:,i);
        X = minimize_L1_proj_subgrad(D, temp);
        X = X/sqrt(sum(X.^2));
        res = predict(classifier, X');
        y_prediction(i) = str2num(res{1,1});
    end
