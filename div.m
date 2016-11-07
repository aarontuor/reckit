function AdivB = div(A, B)
	[m, n] = size(A);
  	[q, r, s] = find(A);	
	[t, u, v] = find(B);
	if all(q == t) && all(r == u)	
		val = s./v;
		AdivB = sparse(q, r, val, m, n);
	else
		AdivB = 0;
	end
end
