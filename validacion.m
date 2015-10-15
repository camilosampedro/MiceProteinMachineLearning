% Validacion modelos
%  Validacion cruzada.
%% Parametros de validacion
% Numero de subconjuntos
%k = 50;
% Cantidad de muestras
[numeroMuestras,~] = size(Y);
% Cantidad de iteraciones
numeroIteraciones = 100;

numeroClases=length(unique(Y));

EficienciaTestGMM=zeros(1,numeroIteraciones);
EficienciaTestRF=zeros(1,numeroIteraciones);
EficienciaTestKNN=zeros(1,numeroIteraciones);
EficienciaTestRNA=zeros(1,numeroIteraciones);

%Yestimado = sim(net,Xtest');

%% Iteraciones
for i=1:numeroIteraciones
    disp('');
    
    rng('default');

    particion=cvpartition(numeroMuestras,'Kfold',numeroIteraciones);
    indices=particion.training(i);
    Xentrenamiento=X(particion.training(i),:);
    Xvalidacion=X(particion.test(i),:);
    Yentrenamiento=Y(particion.training(i));
    Yvalidacion=Y(particion.test(i));

    [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
    Xvalidacion=(Xvalidacion - repmat(mu,size(Xvalidacion,1),1))./repmat(sigma,size(Xvalidacion,1),1);

    % Seleccion aleatoria de los indices
    %indices = randperm(N);
    %Xvalidacion = X(indices(1:N/k),:);
    %Yvalidacion = Y(indices(1:N/k),:);
    %Xentrenamiento = X(indices(N/k + 1: end),:);
    %Yentrenamiento = Y(indices(N/k + 1: end),:);
    
    %patternnet
    %train
    %sim
    
    %fitnet

    %% Modelo de mezclas gaussianas
    % Entrenamiento modelo de mezclas gaussianas
    Texto=['Iteración GMM ', num2str(i),' / ',num2str(numeroIteraciones)];
    disp(Texto);
    
    mezclas = 3;
    modelos = [];
    for j=1:numeroClases
        Yclaseactual = (Yentrenamiento==j);
        Xclaseactual = Xentrenamiento(Yclaseactual,:);
        if ~isempty(Xclaseactual)
            modelos = [modelos,entrenarGMM(Xclaseactual,mezclas,'spherical')];
        else
            error('GMM: No hay muestras para todas las clases');
        end
    end
    % Valicacion modelo de mezclas gaussianas
    resultados = [];
    for j=1:numeroClases
        resultados = [resultados,testGMM(modelos(j),Xvalidacion)];
    end
    [~,Yestimado] = max(resultados,[],2);

    % Matriz de confusion modelo de mezclas gaussianas
    EficienciaTestGMM(i) = calcularEficienciaMC(numeroClases,Xvalidacion,Yestimado,Yvalidacion);

    %% Modelo Random Forest
    Texto=['Iteración RF  ', num2str(i),' / ',num2str(numeroIteraciones)];
    disp(Texto);
    numeroArboles=100;
    modelo=entrenarFOREST(numeroArboles,Xentrenamiento,Yentrenamiento);

    % Validacion
    Yestimado = testFOREST(modelo,Xvalidacion);

    % Matriz de confusion modelo Random Forest
    EficienciaTestRF(i) = calcularEficienciaMC(numeroClases,Xvalidacion,Yestimado,Yvalidacion);

    %% Modelo de k vecinos cercanos
    Texto=['Iteración KNN ', num2str(i),' / ',num2str(numeroIteraciones)];
    disp(Texto);
    kVecinos = 20;
    Yestimado = vecinosCercanos(Xvalidacion,Xentrenamiento,Yentrenamiento,kVecinos,'class');

    % Matriz de confusion modelo Random Forest
    
    EficienciaTestKNN(i) = calcularEficienciaMC(numeroClases,Xvalidacion,Yestimado,Yvalidacion);
    
    
    %% Modelo de red neuronal
    Texto=['Iteración RNA ', num2str(i),' / ',num2str(numeroIteraciones)];
    disp(Texto);
    Yred = zeros(size(Yentrenamiento,1),numeroClases);
    for k=1:size(Yentrenamiento,1)
        Yred(k,Yentrenamiento(k)) = 1;
    end
    capas=[128,64,32,16];
    net = patternnet(capas);
    net.trainParam.showWindow=false;
    net = train(net,Xentrenamiento',Yred');
    % Matriz de confusion modelo Red neuronal
    
    Yestimadored = sim(net,Xvalidacion');
    
    Yestimadored = Yestimadored';
    %disp(Yestimadored);
    [noUsado,Yestimado] = max(Yestimadored,[],2);
    
    EficienciaTestRNA(i) = calcularEficienciaMC(numeroClases,Xvalidacion,Yestimado,Yvalidacion);
end

% Eficiencia del modelo de mezclas gaussianas
EficienciaGMM = mean(EficienciaTestGMM);
ICGMM = std(EficienciaTestGMM);
disp('======== GMM ==========');
Texto=['La eficiencia obtenida fue = ', num2str(EficienciaGMM),' +- ',num2str(ICGMM)];
disp(Texto);

% Eficiencia del modelo Random Forest
EficienciaRF = mean(EficienciaTestRF);
ICRF = std(EficienciaTestRF);
disp('======== RF ===========');
Texto=['La eficiencia obtenida fue = ', num2str(EficienciaRF),' +- ',num2str(ICRF)];
disp(Texto);

% Eficiencia del modelo K vecinos cercanos
EficienciaKNN = mean(EficienciaTestKNN);
ICKNN = std(EficienciaTestKNN);
disp('======== KNN ===========');
Texto=['La eficiencia obtenida fue = ', num2str(EficienciaKNN),' +- ',num2str(ICKNN)];
disp(Texto);

% Eficiencia del modelo Red Neuronal
EficienciaRNA = mean(EficienciaTestRNA);
ICRNA = std(EficienciaTestRNA);
disp('======== RNA ===========');
Texto=['La eficiencia obtenida fue = ', num2str(EficienciaRNA),' +- ',num2str(ICRNA)];
disp(Texto);