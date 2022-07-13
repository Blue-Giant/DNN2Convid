# COVID_DNN
## Requirement
The codes are implemented in tensorflow--1.14 or 1.15 under the interpreter python3.6 or python3.7.  Additionally, if the codes are runned on a Server, one should use the miniconda3 for python 3.7 or 3.6. However, if you dowmload the latest version of miniconda3 from https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh, you will get a miniconda3 based on python 3.8.  Hence, you should redirect to the https://docs.conda.io/en/latest/miniconda.html, then download the miniconda3 based on python3.7.

## Reference paper and URLs:
0、Simulating the spread of COVID-19 via a spatially-resolved susceptible–exposed–infected–recovered–deceased (SEIRD) model with heterogeneous diffusion
https://sciencedirect.53yu.com/science/article/pii/S0893965920303013

1、Analysis of COVID-19 spread in South Korea using the SIR model with time-dependent parameters and deep learning https://www.medrxiv.org/content/10.1101/2020.04.13.20063412v1

2、【数学建模】传染病SIR模型 https://blog.csdn.net/qq_45654306/article/details/108135965

3、传染病SIR模型 https://blog.csdn.net/qq_43585318/article/details/104194312

4、h《全网首发》基于SEIR（SIR）对新冠肺炎的分析和预测 https://wxw-123.blog.csdn.net/article/details/118309560

5、新冠数据整理和简单分析（二）——SIR及其变种 https://blog.csdn.net/weixin_41677876/article/details/105365496

6、新型冠状病毒传染模型SI、SIS、SIR、SEIR https://blog.csdn.net/eddsadsadasd/article/details/104751112

7、2020年美国新冠肺炎疫情数据分析 https://hejianing.blog.csdn.net/article/details/117928134

8、python实现SI、SIS、SIR、SIRS、SEIR、SEIRS模型 https://blog.csdn.net/weixin_43289424/article/details/104214637

9、SIR模型和Python实现 https://blog.csdn.net/weixin_41168304/article/details/122398129

## Euler method

欧拉法及其他改进方法——Matlab实现 https://blog.csdn.net/qq_36312878/article/details/80945781

## Runge-Kutta method

Matlab中龙格-库塔(Runge-Kutta)方法原理及实现 https://blog.csdn.net/xiaokun19870825/article/details/78763739

%参数表顺序依次是微分方程组的函数名称，初始值向量，步长，时间起点，时间终点（参数形式参考了ode45函数）
function [x,y]=runge_kutta(ufunc,y0,h,a,b)
n=floor((b-a)/h);       %步数
x(1)=a;                 %时间起点
y(:,1)=y0;              %赋初值，可以是向量，但是要注意维数
for i=1:n               %龙格库塔方法进行数值求解    
    x(i+1)=x(i)+h;    
    k1=ufunc(x(i),y(:,i));  
    k2=ufunc(x(i)+h/2,y(:,i)+h*k1/2);    
    k3=ufunc(x(i)+h/2,y(:,i)+h*k2/2);   
    k4=ufunc(x(i)+h,y(:,i)+h*k3);   
    y(:,i+1)=y(:,i)+h*(k1+2*k2+2*k3+k4)/6;
end
