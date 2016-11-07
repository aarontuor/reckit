function [U, P] = batchUpdate(Rdif, U, P)
    k = size(U, 1);
    U = U + alpha*((Rdif*P - lambda*U));
    if bias U(:, k+ 2) = 1; end
    P = P + alpha*(smult(R', P, U)*U - lambda*P);
    if bias P(:, k+1) = 1; end
end