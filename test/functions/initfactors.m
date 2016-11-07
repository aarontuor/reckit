function [U, P] = initfactors(m, n, k, update)
    if strcmp(update, 'mult')
        U = rand(m, k);
        P = rand(n, k);
    else
        U = 2*rand(m, k) - 1;
        P = 2*rand(n, k) - 1;
    end
end