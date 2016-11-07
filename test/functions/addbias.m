function [R, Dev, U, P, mu] = addbias(R, Dev, U, P)
    
    m = size(R, 1);
    n = size(R, 2);
    %center R		
    [i, j, v] = find(R);
    %v = v -1;        
    mu = sum(v)/length(v);
    v = v - mu;
    R = sparse(i,j,v, m, n);
            
    %center Dev		
    [i2, j2, v2] = find(Dev);
    %v2 = v2 -1;        
    v2 = v2 - mu;
    Dev = sparse(i2,j2,v2, m, n);

    %add bias to matrices
    userBias = 2 * rand(m,1) - 1;
    userMult = ones(m, 1);
    U = [U userBias userMult];
    itemBias = 2 * rand(n,1) - 1;
    itemMult = ones(n, 1);
    P = [P itemMult itemBias];
end