function AB_T = smult(mask, A, B)
    [m, n] = size(mask);
    [i,j,v] = find(mask);		
    indices = 1:length(j);	
    Arows = A(i(indices), :);
    Brows = B(j(indices), :);
    AB_T_values = dot(Arows, Brows, 2);		
    AB_T = sparse(i, j, AB_T_values, m, n);
end