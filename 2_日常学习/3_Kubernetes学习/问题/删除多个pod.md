```
kubectl get pod -n default | grep Evicted | awk '{print $1}' | xargs kubectl delete pod -n default
```