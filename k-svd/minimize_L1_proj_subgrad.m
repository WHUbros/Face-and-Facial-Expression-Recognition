function x = minimize_L1_proj_subgrad( A, y ) 
% From Professor John Wright
BETA = 1/50;
MAX_ITER = 10000;

[m,n] = size(A);

H = A' * pinv( A * A' );

numIter = 0;
done    = false;

x = zeros(n,1);
k = 1;

allObj = zeros(MAX_ITER,1);

while ~done, 
    
    x_til = x - (BETA/sqrt(k)) * sign(x);            
    x     = x_til - H * ( A * x_til - y );
    
    obj = norm(x,1);    
    allObj(k) = obj;    
        
    k = k + 1;
    
    if k > MAX_ITER, 
        done = true; 
    end    
end
