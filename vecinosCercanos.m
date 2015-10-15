function Yesti = vecinosCercanos(Xval,Xent,Yent,k,tipo)

    %%% El parametro 'tipo' es el tipo de problema que se va a resolver
    
    %%% La función debe retornar el valor de predicción Yesti para cada una de 
    %%% las muestras en Xval. Por esa razón Yesti se inicializa como un vectores 
    %%% de ceros, de dimensión M.

    N=size(Xent,1);
    M=size(Xval,1);
    
    Yesti=zeros(M,1);
    dis=zeros(N,1);

    if strcmp(tipo,'class')
        for j=1:M
            
            %%% Complete el codigo %%%
            for i=1:N
                dis(i)=norm(Xval(j,:) - Xent(i,:));
               
            end
            [a,b]= sort(dis);
            vecinos=b(1:k);
            y=zeros(k,1);
            for z=1:k
                y(z)=Yent(vecinos(z));
            end
            Yesti(j) = mode(y);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        
    elseif strcmp(tipo,'regress')
        
        for j=1:M
            %%% Complete el codigo %%%
            for i=1:N
                dis(i)=norm(Xval(j,:) - Xent(i,:));
               
            end
            [a,b]= sort(dis);
            vecinos=b(1:k);
            y=zeros(k,1);
            for z=1:k
                y(z)=Yent(vecinos(z));
            end
            Yesti(j) = mean(y);
			%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
        
        
        
    end

end