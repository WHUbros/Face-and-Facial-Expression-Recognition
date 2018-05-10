function [D_updated, X_updated] = dic_update(D, X, data, i)
% Update the ith column in the dictionary
% INPUT:
% D: dictionary
% X: sparse coding matrix
% data: data matrix
% i: the ith column in the dictionary
% OUTPUT:
% D_updated: updated dictionary
% X_updated: updated sparse coding
    indexes = find(X(i,:) ~= 0);
    D_updated = D; % d*k
    X_updated = X; % k*m
    
    if length(indexes) >0
        D_updated(:,i) = 0;
        matrix_e_k = data(:, indexes) - D_updated*X_updated(:, indexes);
        % use svd to gaurantee the orthogonality
        [u,s,v] = svds(matrix_e_k, 1);
        D_updated(:, i) = u(:, 1);
        X_updated(i, indexes) = v*s(1,:);
    end
end