function psi = HarMo_Sequence(t, d)
i = (1:d).';
psi = sin((i*pi/(d+1)) * t);
end
