

Xu = [];
for i = 1:size(Xtraining,1)
    if all(Xtraining(i,3:4)==0.06)
        Xu = [Xu; Xtraining(i,:)];
    end
end

Xd = [];
for i = 1:size(Xtraining,1)
    if all(Xtraining(i,3:4)==-0.06)
        Xd = [Xd; Xtraining(i,:)];
    end
end

Xr = [];
for i = 1:size(Xtraining,1)
    if Xtraining(i,3)==-0.06 && Xtraining(i,4)==0.06
        Xr = [Xr; Xtraining(i,:)];
    end
end

Xl = [];
for i = 1:size(Xtraining,1)
    if Xtraining(i,3)==0.06 && Xtraining(i,4)==-0.06
        Xl = [Xl; Xtraining(i,:)];
    end
end


%%
[sum(Xu(:,2)<Xu(:,6)) sum(Xu(:,2)>Xu(:,6)) mean(Xu(:,6)-Xu(:,2))]
[sum(Xd(:,2)<Xd(:,6)) sum(Xd(:,2)>Xd(:,6)) mean(Xd(:,6)-Xd(:,2))]
[sum(Xr(:,1)<Xr(:,5)) sum(Xr(:,1)>Xr(:,5)) mean(Xr(:,5)-Xr(:,1))]
[sum(Xl(:,1)<Xl(:,5)) sum(Xl(:,1)>Xl(:,5)) mean(Xl(:,5)-Xl(:,1))]