# Fold report — time split (expanding), k=5

## Size and coverage

```
fold,split,rows,periods,locations,first_period,last_period,missing
0,train,80,20,4,2000-01,2001-08,8
0,test,80,20,4,2001-09,2003-04,0
1,train,160,40,4,2000-01,2003-04,8
1,test,80,20,4,2003-05,2004-12,0
2,train,240,60,4,2000-01,2004-12,8
2,test,80,20,4,2005-01,2006-08,0
3,train,320,80,4,2000-01,2006-08,8
3,test,80,20,4,2006-09,2008-04,0
4,train,400,100,4,2000-01,2008-04,8
4,test,80,20,4,2008-05,2009-12,0
```

## Planted features per fold

This scenario plants no countable features — no seasonal peaks,
outbreak shocks or declared events.

## Warnings

None — every fold tests every kind of planted feature.
