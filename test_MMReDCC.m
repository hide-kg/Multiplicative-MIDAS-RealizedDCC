load('vechRMK_63_20142020.mat')
for t = 1:1624
    RMK(:,:,t) = ivech(MX(:,t));
end

%stock = [1 5 10 15 20 25];
stock = [15 20];
RC = RMK(stock, stock, :);
L = 240;
test_start = 1624/2;
[estimpara_mm, forecast_fit_mm, logL_mm] = MMReDCC_onestep(RC, L, test_start);
S_mm = forecast_fit_mm.covariance;
[loss_MM_stein, losses_MM_stein] = Stein_loss(RC(:,:,test_start+1:end), S_mm(:,:,test_start+1:end)); 




for t = 241:1624
    RV_fore(:,t) = diag(forecast_fit_mm.covariance(:,:,t));
    RV_obs(:,t) = diag(RC(:,:,t));
end
figure
plot(RV_obs(1,:),'b')
hold on
plot(RV_fore(1,:), 'r', 'LineWidth', 1.5)

figure
plot(RV_obs(2,:),'b')
hold on
plot(RV_fore(2,:), 'r', 'LineWidth', 1.5)
