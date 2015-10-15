function Eficiencia = calcularEficienciaMC(numeroClases,Xvalidacion,Yestimado,Yvalidacion)
    MatrizConfusion=zeros(numeroClases,numeroClases);
    for j=1:size(Xvalidacion,1)
        MatrizConfusion(Yestimado(j),Yvalidacion(j)) = MatrizConfusion(Yestimado(j),Yvalidacion(j))+1;
    end
    Eficiencia = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
end
