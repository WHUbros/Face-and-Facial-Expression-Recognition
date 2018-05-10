function [D, X] = k_svd(data, k)
% Use k-svd to create a dictionary
% INPUT:
% data: original data
% k: number of atoms in a dictionary
% OUTPUT:
% D: dictionary
% X: sparse coding
    [d,n] = size(data);
    % initialize the dictionary
    index = randperm(n);
    index = index(1:k);
    D = data(:, index);
    % initializa the sparse coding
    X = zeros(k, n);
    done = false;
    iter = 0;
    while ~done & iter < 30
        % update the dictionary and sparese representation iterately 
        X = OMP(data, D,k);
        for i = 1:k
            [D, X] = dic_update(D, X, data, i);
        end
        iter = iter + 1;
    end

    for i = 1:n
        X(:, i) = X(:, i)/sqrt(sum(X(:,i).^2));
    end