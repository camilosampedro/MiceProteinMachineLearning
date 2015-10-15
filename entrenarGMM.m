function modelo = entrenarGMM(X,NumeroMezclas,tipo)

    inputDim=size(X,2);      %%%%% Numero de caracteristicas de las muestras
    MIX = gmm(inputDim,NumeroMezclas,tipo);
    opciones = foptions;
    %opciones(14)=10;
    MIX = gmminit(MIX,X,opciones);
    [MIX, opciones, log] = gmmem (MIX, X,opciones);
    modelo = MIX;
end