function [U, P] = initfactors(m, n, k, sign)
    if strcmp(sign, 'plus')
        U = rand(m, k);
        P = rand(n, k);
    elseif strcmp(sign, 'both')
        U = 2*rand(m, k) - 1;
        P = 2*rand(n, k) - 1;
    elseif strcmp(sign, 'minus')
		U = -1*rand(m, k);
        P = -1*rand(n, k);
	end
end
