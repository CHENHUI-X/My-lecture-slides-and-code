%%������
function GA() 

clear
close all
popsize=20; % Ⱥ���С
chromlength=20; %���ĳ��ȣ����峤�ȣ�
pc=0.6; %�������
pm=0.1; %�������
xlim = [0,50];
G = 100 ; %��������
% x = zeros(1,G); % ��¼ÿ����������λ�� 
% y = zeros(1,G); % ��¼ÿ�����Ÿ����Ӧ�ĺ���ֵ

pop= round( rand(popsize,chromlength) )  ; %���������ʼȺ��
decpop =  bintodec( pop ,popsize, chromlength,xlim ) ; % ����������Ӧʮ����
fx = calobjvalue(decpop  ) ; % ���������ĺ���ֵ
plotfig(decpop , fx , xlim , 1  ) ; % ����ͼ��
[y(1) , l ]  = min(fx); x(1) = decpop(l);

for i=2  : G 
    decpop =  bintodec( pop , popsize, chromlength,xlim ) ; % ������һ�����Ӧʮ����
    fx = calobjvalue(decpop  ) ; % ������һ����ĺ���ֵ
    fitvalue = calfitvalue(fx) ;  % ��Ӧ��ӳ��
    newpop = copyx(pop,fitvalue,popsize); %����
    newpop = crossover(newpop, pc, popsize,chromlength ); %����
    newpop = mutation(newpop,pm, popsize,chromlength); %����
    % ��ʱ��newpop�Ǿ������ƽ�������������һ��Ⱥ��
    %     �±߽���ѡ�����ű�������ʵ�ֱ��׻��ƣ�
    newdecpop =  bintodec( newpop ,popsize, chromlength,xlim ) ;
    new_fx = calobjvalue(newdecpop) ; %�����½�Ŀ�꺯��
    new_fitvalue = calfitvalue(new_fx); %������Ⱥ����ÿ���������Ӧ��
    index = find(new_fitvalue > fitvalue) ; 
    
    pop(index, : ) = newpop(index,:) ; % ���µõ����½�
    decpop = bintodec( pop ,popsize, chromlength,xlim ) ; %�����½��ʮ����
    fx = calobjvalue( decpop )  ; %������
    plotfig(decpop , fx ,xlim , i ) % �����½��ͼ
    % �ҳ����º�ĸ������ź���ֵ
    [bestindividual,bestindex] = max(  fx ) ;
    y(i)=bestindividual; % ��¼ÿһ�������ź���ֵ
    x(i)= decpop(bestindex) ; %ʮ���ƽ�
    subplot(1,2,2);
    plot(1:i,y); 
    title('��Ӧ�Ƚ�������');
    i = i + 1 ;
end
[ymax, max_index] = max(y);
disp(['�ҵ����Ž�λ��Ϊ��', num2str(x(max_index)) ])
disp(['��Ӧ���Ž�Ϊ��', num2str(ymax) ])

end

%******************************************************************************************%
%% ������Ӧ��
function fitvalue = calfitvalue(fx)
%���������ֵ�����Һ���ֵ�ֶ�����0������ֱ��ʹ�ú���ֵ������Ϊ��Ӧ��ֵ��
% ��ʵ�ϣ���ͬ��������Ӧ�Ⱥ������췽�����ֶ�����
    fitvalue = fx ; 
end

%% ���Ʋ���
function newx = copyx(pop, fitvalue,popsize ) %�����������ƴ��Ͷ�Ӧ��Ӧ��
% ����PPT�����̶Ĳ��ԶԸ��帴��
    newx = pop; %ֻ��������һ��sizeΪpop��С�ռ�����ã�newx֮��Ҫ���µ�
    i = 1;  j = 1;
    p = fitvalue / sum(fitvalue) ; 
    Cs = cumsum(p) ; 
    R = sort(rand(popsize,1)) ; %ÿ������ĸ��Ƹ���
    while j <= popsize 
        if R(j) < Cs(i)
            newx(j,:) = pop(i,:) ;
            j = j + 1;
        else
            i = i + 1;
        end
    end
end

%% �������
function newx = crossover(pop, pc, popsize,chromlength )
% 12 34 56���淽ʽ�����ѡ�񽻲�λ��
% ע�������Ϊ����ż��������
i = 2 ;
newx = pop ; %����ռ�
while i + 2 <= popsize
    %����i �� �� i -1 �������λ�㽻��
    if rand < pc
        x1 = pop(i-1,:);
        x2 = pop(i,:) ; 
        r = randperm( chromlength , 2 ) ; %���ط�Χ����������
        r1 = min(r); r2 =max(r) ; % ���渴�Ƶ�λ��
        newx(i-1,:) = [x1( 1 : r1-1),x2(r1:r2) , x1(r2+1: end)];
        newx(i , : ) = [x2( 1 : r1-1),x1(r1:r2) , x2(r2+1: end)];
    end
    i = i + 2 ; %����i
end

end

%% ����
function newx = mutation(pop,pm, popsize,chromlength)
i = 1 ;
while i <= popsize
    if rand < pm
        r = randperm( chromlength , 1 ) ; 
        pop(i , r) = ~pop(i, r);
    end
    i = i + 1;
end

newx = pop; %�������Ľ�����ء�

end

%%  ������תʮ���ƺ���
function dec = bintodec( pop ,popsize, chromlength,xlim )
    dec = zeros(1,chromlength);
    index = chromlength-1:-1:0;
    for i = 1 : popsize
        dec(i) = sum(pop(i,:).* (2.^index));
    end
    dec = xlim(1) + dec*(  xlim(2) -  xlim(1) ) /( 2 ^ chromlength - 1) ;
end


%%  ����ͼ��
function plotfig(decpop , fx ,xlim,k) 
    f = @(x) abs(x .* sin(x) .* cos(2 * x) - 2 * x .* sin(3 * x) +3 * x .* sin(4 * x)); % �о�������  
    x = xlim(1):0.05:xlim(2);
    y = f(x) ; 
    subplot(1,2,1);
    plot(x,y,decpop,fx,'o')
    title(['��',num2str(k),'�ε�������'])
    pause(0.2)
end


%% Ŀ�꺯��
function fx = calobjvalue(decpop ) %����Ϊʮ���ƽ�

f = @(x) abs(x .* sin(x) .* cos(2 * x) - 2 * x .* sin(3 * x) +3 * x .* sin(4 * x)) ; % �о�������        
fx = f(decpop);
end








