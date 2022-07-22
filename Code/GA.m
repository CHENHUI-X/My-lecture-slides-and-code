%%主程序
function GA() 

clear
close all
popsize=20; % 群体大小
chromlength=20; %串的长度（个体长度）
pc=0.6; %交叉概率
pm=0.1; %变异概率
xlim = [0,50];
G = 100 ; %迭代次数
% x = zeros(1,G); % 记录每代个体最优位置 
% y = zeros(1,G); % 记录每代最优个体对应的函数值

pop= round( rand(popsize,chromlength) )  ; %随机产生初始群体
decpop =  bintodec( pop ,popsize, chromlength,xlim ) ; % 计算初代解对应十进制
fx = calobjvalue(decpop  ) ; % 计算初代解的函数值
plotfig(decpop , fx , xlim , 1  ) ; % 绘制图像
[y(1) , l ]  = min(fx); x(1) = decpop(l);

for i=2  : G 
    decpop =  bintodec( pop , popsize, chromlength,xlim ) ; % 计算上一代解对应十进制
    fx = calobjvalue(decpop  ) ; % 计算上一代解的函数值
    fitvalue = calfitvalue(fx) ;  % 适应度映射
    newpop = copyx(pop,fitvalue,popsize); %复制
    newpop = crossover(newpop, pc, popsize,chromlength ); %交叉
    newpop = mutation(newpop,pm, popsize,chromlength); %变异
    % 这时的newpop是经过复制交叉变异产生的新一代群体
    %     下边进行选择择优保留（即实现保底机制）
    newdecpop =  bintodec( newpop ,popsize, chromlength,xlim ) ;
    new_fx = calobjvalue(newdecpop) ; %计算新解目标函数
    new_fitvalue = calfitvalue(new_fx); %计算新群体中每个个体的适应度
    index = find(new_fitvalue > fitvalue) ; 
    
    pop(index, : ) = newpop(index,:) ; % 更新得到最新解
    decpop = bintodec( pop ,popsize, chromlength,xlim ) ; %计算新解的十进制
    fx = calobjvalue( decpop )  ; %计算结果
    plotfig(decpop , fx ,xlim , i ) % 绘制新解的图
    % 找出更新后的个体最优函数值
    [bestindividual,bestindex] = max(  fx ) ;
    y(i)=bestindividual; % 记录每一代的最优函数值
    x(i)= decpop(bestindex) ; %十进制解
    subplot(1,2,2);
    plot(1:i,y); 
    title('适应度进化曲线');
    i = i + 1 ;
end
[ymax, max_index] = max(y);
disp(['找的最优解位置为：', num2str(x(max_index)) ])
disp(['对应最优解为：', num2str(ymax) ])

end

%******************************************************************************************%
%% 计算适应度
function fitvalue = calfitvalue(fx)
%这里求最大值，并且函数值又都大于0，所以直接使用函数值本身作为适应度值。
% 事实上，不同的问题适应度函数构造方法多种多样。
    fitvalue = fx ; 
end

%% 复制操作
function newx = copyx(pop, fitvalue,popsize ) %传进来二进制串和对应适应度
% 按照PPT的轮盘赌策略对个体复制
    newx = pop; %只是起到申请一个size为pop大小空间的作用，newx之后要更新的
    i = 1;  j = 1;
    p = fitvalue / sum(fitvalue) ; 
    Cs = cumsum(p) ; 
    R = sort(rand(popsize,1)) ; %每个个体的复制概率
    while j <= popsize 
        if R(j) < Cs(i)
            newx(j,:) = pop(i,:) ;
            j = j + 1;
        else
            i = i + 1;
        end
    end
end

%% 交叉操作
function newx = crossover(pop, pc, popsize,chromlength )
% 12 34 56交叉方式，随机选择交叉位点
% 注意个体数为奇数偶数的区别
i = 2 ;
newx = pop ; %申请空间
while i + 2 <= popsize
    %将第i 与 第 i -1 进行随机位点交叉
    if rand < pc
        x1 = pop(i-1,:);
        x2 = pop(i,:) ; 
        r = randperm( chromlength , 2 ) ; %返回范围内两个整数
        r1 = min(r); r2 =max(r) ; % 交叉复制的位点
        newx(i-1,:) = [x1( 1 : r1-1),x2(r1:r2) , x1(r2+1: end)];
        newx(i , : ) = [x2( 1 : r1-1),x1(r1:r2) , x2(r2+1: end)];
    end
    i = i + 2 ; %更新i
end

end

%% 变异
function newx = mutation(pop,pm, popsize,chromlength)
i = 1 ;
while i <= popsize
    if rand < pm
        r = randperm( chromlength , 1 ) ; 
        pop(i , r) = ~pop(i, r);
    end
    i = i + 1;
end

newx = pop; %将变异后的结果返回。

end

%%  二进制转十进制函数
function dec = bintodec( pop ,popsize, chromlength,xlim )
    dec = zeros(1,chromlength);
    index = chromlength-1:-1:0;
    for i = 1 : popsize
        dec(i) = sum(pop(i,:).* (2.^index));
    end
    dec = xlim(1) + dec*(  xlim(2) -  xlim(1) ) /( 2 ^ chromlength - 1) ;
end


%%  绘制图像
function plotfig(decpop , fx ,xlim,k) 
    f = @(x) abs(x .* sin(x) .* cos(2 * x) - 2 * x .* sin(3 * x) +3 * x .* sin(4 * x)); % 研究对象函数  
    x = xlim(1):0.05:xlim(2);
    y = f(x) ; 
    subplot(1,2,1);
    plot(x,y,decpop,fx,'o')
    title(['第',num2str(k),'次迭代进化'])
    pause(0.2)
end


%% 目标函数
function fx = calobjvalue(decpop ) %参数为十进制解

f = @(x) abs(x .* sin(x) .* cos(2 * x) - 2 * x .* sin(3 * x) +3 * x .* sin(4 * x)) ; % 研究对象函数        
fx = f(decpop);
end








