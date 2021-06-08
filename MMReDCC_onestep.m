
function [estimpara, forecast_fit, logL] = MMReDCC_onestep(RC, L, test_start)
%
% 2021/3/18
%   Bauwens et al. (2016)��ReDCC���f����MIDAS�g���^���f���̎���
%   �{�v���O������Multiplicative�^���f���̐��������
%   ������@��1-�X�e�b�v�^�Ƃ���. 
%   1-�X�e�b�v����͎����̎􂢂ɂ���č������ō���ɂȂ邪, Bauwens et al. (2017)�̐���@�ŉ����ł���
%   
% reference 
%   Bauwens, L., Braione, M. and Storti, G. (2017). A dynamic component
%   model for forecasting high-dimensional realized covariance matrices.
%
% input : 
%   RC - ���������U�s��
%   test_start - �\�����Ԃ̎n�܂�
%
% output : 
%   para_garch - GARCH���̃p�����[�^
%   para_dcc - �����t�����ւ̃p�����[�^

[K, ~, T] = size(RC);

para0 = mmredcc_initpara(K);

%% �p�����[�^����
warning('off') %#ok<*WNOFF>
options = optimoptions('fminunc','Display', 'off');
ll = @(x0) -MMReDCC_llh(x0, RC, L, test_start, 0);
[para] = fminunc(ll, para0, options);
warning('on') %#ok<*WNON>

[gamma, delta, alpha, beta, Delta, theta, omega_r, nu] = MMReDCC_transpara(para, K);

%% �p�����[�^�̐���
% Delta �� n*n �̐���l�s��Ȃ̂�, ��U���O�p�s��Ő��肷��. ���̌㌳�ɖ߂�. 
gamma = gamma.^2;
delta = delta.^2;
alpha = alpha.^2;
beta = beta.^2;
Delta = Delta * Delta';
theta = theta.^2;
nu = abs(nu) + K;
omega_r = abs(omega_r) + 1;

%% �����U�s��̗\���Ƒΐ��ޓx�̎擾
[~, S] = MMReDCC_llh(para, RC, L, test_start, 1);

llh = wishlike(S(:,:,L+2:test_start-1)./nu, nu, RC(:,:,L+2:test_start-1));

%% AIC��BIC�̌v�Z
num_para = 2 * K + 2 + K * (K+1)/2 + 3;
aic = -2 * llh + 2 * num_para;
bic = -2 * llh + log(K * (T-L+1)) * num_para;

estimpara = struct();
estimpara.variance_short = [gamma, delta];
estimpara.correlation_short = [alpha, beta];
estimpara.long_matrix = Delta;
estimpara.long_scalar = [theta ,omega_r];
estimpara.degree_of_free = nu;

forecast_fit = struct();
forecast_fit.covariance = S;

logL = struct();
logL.llh = llh;
logL.AIC = aic;
logL.BIC = bic;

end

%% �ΐ��ޓx�֐�
function [llh, S] = MMReDCC_llh(para0, RC_input, L, test_start, type)
%
% MMReDCC���f���̑ΐ��ޓx�֐��̃v���O����
% input : 
%   type - 0 (�Ŗޖ@�ɂ�����ޓx�̌v�Z)
%          1 (�p�����[�^�����ɑΐ��ޓx�Ƌ����U�̗\���l�̌v�Z)

if type == 0
    RC = RC_input(:,:,1:test_start-1);
elseif type == 1
    RC = RC_input;
end


[K, ~, T] = size(RC);

[gamma, delta, alpha, beta, Delta, theta, omega_r, nu] = MMReDCC_transpara(para0, K);

gamma = gamma.^2;
delta = delta.^2;

alpha = alpha^2;
beta = beta^2;

Delta = Delta * Delta';
theta = theta.^2;
nu = abs(nu) + K;
omega_r = abs(omega_r) + 1;

S = zeros(K, K, T);
for t = 1:L
    S(:,:,t) = mean(RC(:,:,1:t), 3);
end

%% 1.1 ��������
beta_term = zeros(1,1,L);
for l = 1:L
    beta_term(1,1,l) = beta_weight(l, L, omega_r);
end
M = zeros(K,K,T);
Mchol = zeros(K, K, T);
for t = L+1:T
    RC_lag = RC(:,:,t-L:t-1);
    M(:,:,t) = Delta * Delta' + theta * sum(flip(beta_term) .* RC_lag, 3);
    Mchol(:,:,t) = chol(M(:,:,t), 'lower');
end

%% 2.1 �Z�������̏����ݒ�
% 1:L�͒��������̌v�Z�Ŏg���Ă���̂ŒZ�������ł͎g�p�ł��Ȃ�.
% ���̂���, �����_�ƍŏI�_�����߂Ďw�肷��. 
start_time = L+1;
end_time = T;
insample_period = test_start-1;

S_star = zeros(K, K, end_time);
C_star = zeros(K, K, end_time);
D_star = zeros(K, K, end_time);
% 2021/3/19
%   1:L�܂ł�Mchol���v�Z���Ă��Ȃ��̂�, ���̊��Ԃ��܂߂��NaN����������. 
for t = start_time:end_time
    C_star(:,:,t) = Mchol(:,:,t)\RC(:,:,t)/(Mchol(:,:,t)');
end
S(:,:,start_time) = mean(RC(:,:,1:start_time),3);
S_star(:,:,start_time) = Mchol(:,:,start_time)\S(:,:,start_time)/(Mchol(:,:,start_time)');

R_star = zeros(K, K, end_time);
P_star = zeros(K, K, end_time);
S_ii_star = zeros(end_time, K);
C_ii_star = zeros(end_time, K);

[R_star(:,:,start_time), S_ii_star(start_time,:)] = cov_to_corr(S_star(:,:,start_time));
[P_star(:,:,start_time:end_time), C_ii_star(start_time:end_time,:)] = cov_to_corr(C_star(:,:,start_time:end_time));

%% 2.2 �Z�������̕��U�̐���
for k = 1:K
    for t = start_time+1:end_time
        S_ii_star(t,k) = (1 - gamma(k) - delta(k)) ...
            + gamma(k) * C_ii_star(t-1,k) + delta(k) * S_ii_star(t-1,k);
        D_star(k,k,t) = sqrt(S_ii_star(t,k));
    end
end

%% 2.3 �Z�������̑��ւ̐���
for t = start_time+1:end_time
    %% ���֍s��̐���
    R_star(:,:,t) = (1 - alpha - beta) * eye(K) ...
        + alpha * P_star(:,:,t-1) + beta * R_star(:,:,t-1);
    
    %% �Z�������̋����U�̐���
    S_star(:,:,t) = D_star(:,:,t) * R_star(:,:,t) * D_star(:,:,t);
    
    %% �Z�������̋����U�ƒ������������킹�ċ����U�s��𐄒肷��
    S(:,:,t) = Mchol(:,:,t) * S_star(:,:,t) * Mchol(:,:,t)';        
end
if type == 1
    S = insanity_filter(RC, S, start_time+1);
end

llhs = zeros(T, 1);
if type == 0
    for t = start_time+1:end_time
        llhs(t,1) = -nu/2 * (log(det(S(:,:,t))) + trace(S(:,:,t)\RC(:,:,t)));
    end
    llh = sum(llhs(start_time+1:end_time));
elseif type == 1
    for t = start_time+1:insample_period
        llhs(t,1) = -nu/2 * (log(det(S(:,:,t))) + trace(S(:,:,t)\RC(:,:,t)));
    end
    llh = sum(llhs(start_time+1:insample_period));
end


%% �p�����[�^�̊m�F
if type == 0
    sum_para = zeros(K, 1);
    for k = 1:K
        sum_para(k) = gamma(k) + delta(k);
        if sum_para(k) >= 1
            llh = -inf;
        end
        if alpha + beta >= 1
            llh = -inf;
        end
    end
end

end

%% �����p�����[�^
function para = mmredcc_initpara(K)
% MMReDCC���f���̍Ŗޖ@�̂��߂̏����p�����[�^
% �������K�v�ȃp�����[�^��, gamma, delta�̂�
% alpha, beta, theta, omega_r�͈�ŗǂ�
% Delta�͉��O�p�s��
% �p�����[�^���i�[���鏇�Ԃ�[gamma, delta, alpha, beta, Delta, theta, omega_r, nu]
% �����l�͂�����x����������Ă���. 

para_variance_short = K * 2;
para_correlation_short = 2;
para_long = (K*(K+1))/2 + 2;
para_nu = 1;

num_para = para_variance_short + para_correlation_short + para_long + para_nu;

para = ones(num_para,1);

%% �Z�������̃p�����[�^�̏����l
para(1:K) = 0.5 * para(1:K);
para(K+1:2*K) = 0.8 * para(K+1:2*K);
para(2*K+1) = 0.5 * para(2*K+1);
para(2*K+2) = 0.8 * para(2*K+2);

%% Delta�̏����l
weight = rand(K*(K+1)/2,1);
para(2*K+3:2*K+2+(K*(K+1)/2)) = weight .* para(2*K+3:2*K+2+(K*(K+1)/2));

ind = 2*K+2+(K*(K+1)/2);
%% theta��omega_r�̏����l
para(ind+1) = 0.8;
para(ind+2) = 0.5;

para(ind+3) = 1+K;
end

%% �p�����[�^�̌`��ϊ�
function [gamma, delta, alpha, beta, Delta, theta, omega_r, nu] = MMReDCC_transpara(para, K)

gamma = para(1:K, 1);
delta = para(K+1:2*K, 1);
alpha = para(2*K+1,1);
beta = para(2*K+2,1);

Delta_vec = para(2*K+3:2*K+2+(K*(K+1)/2));
Delta = ivech(Delta_vec) - triu(ivech(Delta_vec),1);

ind = 2*K+(K*(K+1)/2);

theta = para(ind+1,1);
omega_r = para(ind+2,1);

nu = para(ind+3,1);

end

%% beta weight
function phi_ell = beta_weight(l, L, omega)
% MIDAS����beta weight
% omega > 1�ł���K�v������
j = 1:L;

phi_ell_upp = (1 - l/L).^(omega-1);
phi_ell_low = sum((1 - j./L).^(omega-1));

phi_ell = phi_ell_upp./phi_ell_low;

end

%% �����U�s�񂩂瑊�֍s���
function [correlation, variance] = cov_to_corr(RC)
% �����U�s�񂩂瑊�֍s���
% �Ŗޖ@�̓r���ŋ��ނ̂�, �s�񂪔�����l�ɂȂ�Ȃ��ꍇ������. 
% output :
%   correlation - K*K*T�s��
%   variance - T*K�s��
 
[K, ~, T] = size(RC);
variance = zeros(T, K);
correlation = zeros(K, K, T);

for t = 1:T
    variance(t,:) = diag(RC(:,:,t));    
    correlation(:,:,t) = sqrt(diag(diag(RC(:,:,t))))\RC(:,:,t)/sqrt(diag(diag(RC(:,:,t))));    
end

end


%% insanity filter
function S_insanity = insanity_filter(RC, S, start)
% input :
%   RC - ���������U(�ϑ��l)
%   S - �\���l

[K, ~, T] = size(RC);
[S_corr, S_variance] = cov_to_corr(S);
[~, RC_variance] = cov_to_corr(RC);
S_insanity = zeros(K, K, T);
M = max(RC_variance);
for t = start:T
    for k = 1:K
        if S_variance(t,k) > M(k)
            S_variance(t,k) = quantile(RC_variance(start:t,k), 0.75);
        end
    end
    S_insanity(:,:,t) = sqrt(diag(S_variance(t,:))) * S_corr(:,:,t) * sqrt(diag(S_variance(t,:)));

end
end

%% �ΐ��ޓx�̌v�Z
function [logL] = wishlike(Sigma, df, data)
% Wishart ���z�̖ޓx�֐�
%   ���R�x�̐���Ƒΐ��ޓx�̌v�Z�Ɏg��
%
% input : 
%   Sigma - Wishart���z�̃X�P�[���s��. ���肵�������U�s������R�x�Ŋ���������
%   df - ���R�x
%   data - ���������U�s��
[K, ~, T] = size(Sigma);
Gamma = 0;
for i = 1:K
    Gamma = Gamma + log(gamma((df + 1 - i)/2));
end

logLs = zeros(T,1);
for t = 1:T
    logLs(t) = -df/2 * log(det(Sigma(:,:,t))) - ...
        1/2 * trace(Sigma(:,:,t)\data(:,:,t)) + ...
        (df - K - 1)/2 * log(det(data(:,:,t)));
end

logLs = logLs - df * K/2 * log(2) - Gamma;

logL = sum(logLs);
end




