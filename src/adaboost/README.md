# AdaBoost

## 先說結論： logitic regression + AdaBoost 是沒有意義的組合

## 血淚史
一開始使用作業的 Decision stump 去跑，但跑太久了，沒辦法跑多 iteration 去比較QQ  
再來改換成 Logistic Regression 去嘗試，發現 AdaBoost 使用 logistic 做分類完全沒有效果QQ  
logistic 部份決定去用 blending...
網路上說 AdaBoost 應該要用有效率的弱分類器  
於是使用 linear regression 來嘗試~
