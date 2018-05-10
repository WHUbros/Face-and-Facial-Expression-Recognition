function [X_train, y_train, break_points]= pre_process(X, y)
    [y_train, index] = sort(y);
    % X: d*n
    X_train = X(:, index);
    break_points = zeros(1,10);
    for i = 1:10
        temp = find(y_train==i-1);
        break_points(i) = temp(1);
    end

    