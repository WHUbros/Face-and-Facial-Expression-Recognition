function [ X ] = OMP( data, D, canshu)
% Update the sparse coding of original data
% INPUT:
% data :original data matrix
% D: dictionary
% sparsity: in range(0, 1]
% OUTPUT:
% X: new sparse coding of original data
[~,n] = size(data);
[~,k] = size(D);
X = zeros(k,n);
% number of nonzero elements
% non_zero = int32(sparsity*k);
for i=1:1:n
    a = [];
    pos_a = zeros(k,1);
    x = data(:,i);
    res = x;
    for j=1:1:canshu
        product = D'*res;
        [~,pos] = max(abs(product));
        pos = pos(1);
        pos_a(j) = pos;
        a = pinv(D(:,pos_a(1:j)))*x;
        res = x - D(:,pos_a(1:j))*a;
        if sum(res.^2)<1e-3
            break;
        end
    end
    temp = zeros(k,1);
    temp(pos_a(1:j)) = a;
    X(:,i) = sparse(temp);
end

end

